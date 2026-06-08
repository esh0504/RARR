import os
from config import Config 

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models import find_models_def
from losses import find_loss_def
from data import find_datasets_def

import argparse
import subprocess
import wandb

import random
import time
import numpy as np

import utils
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
import kornia

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--config', '--cfg', default='Configs/train.yml')

def INIT_ENV(opt):
    # GPU Controller
    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = opt.CUDA_DEVICE_ORDER
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Seed Controller
    random.seed(opt.SEED)
    np.random.seed(opt.SEED)
    torch.manual_seed(opt.SEED)
    torch.cuda.manual_seed_all(opt.SEED)

def INIT_WANDB(opt):
    wandb.login(key=opt.WANDB.APIKEY)

    wandb.init(
        project="RainRemover",
        config = {
            "seed": opt.SEED,
            "architecture": opt.MODEL.SESSION,
            "datasets": opt.DATASET.TRAIN.DATADIR,
            "batch_size": opt.OPTIM.BATCH_SIZE,
            "patch_size": opt.DATASET.TRAIN.PS,
            "loss": opt.LOSS,
            "learing_rate": opt.OPTIM.LR_INITIAL
        }
    )

    wandb.run.name = opt.EXP_NAME + '_' + opt.WANDB.USER

    print('------------------------------------------------------------------------------')
    print("LOGIN WANDB")
    print('------------------------------------------------------------------------------')


if __name__ == '__main__':

    args = parser.parse_args()
    opt = Config(args.config)

    INIT_ENV(opt)
    if opt.WANDB.USE:
        INIT_WANDB(opt)
        

        
    mode = opt.MODEL.MODE
    model_name = opt.MODEL.SESSION

    save_dir = os.path.join(opt.SAVE.DIR, opt.EXP_NAME)
    utils.mkdir(save_dir)

    result_dir = os.path.join(save_dir, mode, 'results', model_name)
    model_dir  = os.path.join(save_dir, mode, 'models',  model_name)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = opt.DATASET.TRAIN.DATADIR
    val_dir   = opt.DATASET.VALIDATE.DATADIR

    ######### Model ###########
    model_restoration = find_models_def(model_name)()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    start_epoch = 1

    ######### Loss ###########
    criterions = {find_loss_def(loss)().cuda(): weight for loss, weight in opt.LOSS[0].items()}

    loss_params = []
    for crit in criterions.keys():
        for p in crit.parameters():
            if p.requires_grad:
                loss_params.append(p)

    if len(loss_params) > 0:
        optimizer.add_param_group({'params': loss_params})

    ######### Resume ###########
    if opt.DATASET.TRAIN.RESUME:
        path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids)>1:
        model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

    ######### DataLoaders ###########
    train_dataset = find_datasets_def('RainTrain')(train_dir,{'patch_size':opt.DATASET.TRAIN.PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

    
    val_dataset = find_datasets_def('RainVal')(val_dir,{'patch_size':opt.DATASET.VALIDATE.PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    epoch_time = []

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()
            restored = model_restoration(input_)
            losses = []

            if model_name != 'MPRNet':
                target = kornia.geometry.transform.build_pyramid(target, 3)
                for loss, weight  in criterions.items():
                    losses.append(weight * torch.sum(torch.stack([loss(restored[j], target[j]) for j in range(len(restored))])))
            else:
                for loss, weight  in criterions.items():
                    losses.append(weight * torch.sum(torch.stack([loss(restored[j], target) for j in range(len(restored))])))

            
            loss = sum(losses)
            
            loss.backward()
            optimizer.step()
            epoch_loss +=loss.item()

        #### Evaluation ####
        if epoch%opt.DATASET.VALIDATE.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)
                restored = restored[0]

                for res,tar in zip(restored,target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch, 
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))

            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 
        
        if opt.WANDB.USE:
            log = {}
            for key, losses in zip(opt.LOSS[0].keys(), losses):
                log[key] = losses
            log['total_loss'] = loss
            if epoch%opt.DATASET.VALIDATE.VAL_AFTER_EVERY == 0:
                log['psnr'] = psnr_val_rgb

            wandb.log(log)


        scheduler.step()
        cur_epoch_time = time.time()-epoch_start_time
        epoch_time.append(cur_epoch_time)
        
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, cur_epoch_time, epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth")) 
    print("Total epochs time : {:.4f}\tAverage epoch time : {:.4f}".format(sum(epoch_time), sum(epoch_time)/len(epoch_time)))
    if opt.WANDB.USE:
        wandb.finish()
