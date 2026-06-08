"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
from config import Config 

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data import find_datasets_def
from models import find_models_def
from skimage import img_as_ubyte
from pdb import set_trace as stx

from evaluate import *

def INIT_ENV(opt):
    # GPU Controller
    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = opt.CUDA_DEVICE_ORDER
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


parser = argparse.ArgumentParser(description='Image Deraining')
parser.add_argument('--config', '--cfg', default='Configs/test.yml')

if __name__=='__main__':
    args = parser.parse_args()
    opt = Config(args.config)

    INIT_ENV(opt)

    model = opt.MODEL.MODE
    model_name = opt.MODEL.SESSION
    model_restoration = find_models_def(model_name)()
    checkpoint_path = os.path.join(opt.ROOT.DIR, opt.EXP_NAME, opt.MODEL.MODE, 'models', model_name, opt.ROOT.WEIGHT)

    utils.load_checkpoint(model_restoration, checkpoint_path)
    print("===>Testing using weights: ", checkpoint_path)
    model_restoration.cuda()
    model_restoration.eval()

    datasets = opt.DATASET.TEST[0].keys()

    for dataset in datasets:
        dataname = dataset.split('/')
        dataname = dataname[-1] if dataname[-1]!='test' else dataname[-2]
        result_dir = os.path.join(opt.ROOT.DIR, opt.EXP_NAME, opt.MODEL.MODE, 'results', model_name, dataname)
        utils.mkdir(result_dir)
        dataset = opt.DATASET.TEST[0][dataset]

        rgb_dir_test = os.path.join(dataset, 'input')
        test_dataset = find_datasets_def('RainTest')(rgb_dir_test,img_options={})

        test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

        with torch.no_grad():
            for ii, data_test in enumerate(tqdm(test_loader), 0):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                input_    = data_test[0].cuda()
                filenames = data_test[1]
                height, width = input_.shape[2], input_.shape[3]
                img_multiple_of = 16
                H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
                padh = H-height if height%img_multiple_of!=0 else 0
                padw = W-width if width%img_multiple_of!=0 else 0
                input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                try:
                    restored = model_restoration(input_)
                except:
                    print(filenames)
                restored = torch.clamp(restored[0],0,1)

                # unpad
                restored = restored[:,:,:height,:width]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(restored)):
                    restored_img = img_as_ubyte(restored[batch])
                    utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
    
    total_psnr = 0
    total_ssim = 0
    num_datasets = len(datasets)
    for dataset in datasets:
        dataname = dataset.split('/')
        dataset_N = opt.DATASET.TEST[0][dataset]
        dataname = dataname[-1] if dataname[-1]!='test' and dataname[-1]!="" else dataname[-2]
        # try:
        qm_psnr, qm_ssim = process_dataset(dataset_N, result_dir, model_name, dataname)
        # except:
        #     print(dataset)
        total_psnr += qm_psnr
        total_ssim += qm_ssim

    print(f'For all datasets PSNR: {total_psnr / num_datasets:.4f} SSIM: {total_ssim / num_datasets:.4f}')