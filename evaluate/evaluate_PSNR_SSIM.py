import os
import numpy as np
import re
import cv2
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool
from functools import partial

def extract_number(filename):
    match = re.search(r'\d+', os.path.basename(filename))
    return int(match.group()) if match else 0

def compute_ssim(img1, img2):
    if img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def compute_psnr_rgb(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')
    psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
    
    return psnr_value
def compute_psnr(img1, img2):
    if img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    if img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def process_image(image_path, gt_path):
    input_img = cv2.imread(image_path)
    gt_img = cv2.imread(gt_path)
    try:
      psnr_val = compute_psnr(input_img, gt_img)
    #   psnr_val = compute_psnr_rgb(input_img, gt_img)
      ssim_val = compute_ssim(input_img, gt_img)
    except:
        print(image_path, gt_path)
    return psnr_val, ssim_val

def match_files(image_files, norain_dir):
    gt_files = []
    temp_files = [x.split('/')[-1] for x in image_files]
    for tmp_file in temp_files:
        if tmp_file.split('-')[0]=='pie':
            target_file = tmp_file.replace("pie-rain-", "pie-norain-")
        else:
            target_file = tmp_file.replace("rain-", "norain-")
        
        gt_file_path = os.path.join(norain_dir, target_file)
        print(tmp_file.split('/')[-1], gt_file_path.split('/')[-1])
        gt_files.append(gt_file_path)

    return gt_files

def process_dataset(dataset, result_dir, model_name, dataname):
    
    file_path = result_dir
    gt_path = f'{dataset}/target/'
    image_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(('.jpg', '.png'))]
    img_num = len(image_files)
   
    if dataname=="DID-MDN-test" or dataname=="Rain1400" or dataname=="RainDS_real":
        image_files = sorted(image_files, key=extract_number)
        gt_files = [os.path.join(gt_path, f) for f in os.listdir(gt_path) if f.endswith(('.jpg', '.png'))]
        gt_files = sorted(gt_files, key=extract_number)
    elif dataname=="RainDS_syn":
        gt_files = match_files(image_files, norain_dir = "Datasets/RainDS_syn/test/target")
    else:
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        gt_files = [os.path.join(gt_path, f) for f in os.listdir(gt_path) if f.endswith(('.jpg', '.png'))]
        gt_files = sorted(gt_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))


    if img_num > 0:
        with Pool() as pool:
            results = pool.starmap(process_image, zip(image_files, gt_files))
        max_r, max_i = 0, 0
        for i,result in enumerate(results):
            print(result[0])
            if max_r<result[0]:
                max_r = result[0]
                max_i = i
        print(max_i, image_files[max_i])
        total_psnr = sum([res[0] for res in results])
        total_ssim = sum([res[1] for res in results])
        
        qm_psnr = total_psnr / img_num
        qm_ssim = total_ssim / img_num
        
        if dataset=="Rain200H/Rain200H/test":
            print(f'For Rain200H dataset PSNR: {qm_psnr:.4f} SSIM: {qm_ssim:.4f}')
        elif dataset=="Rain200L/Rain200L/test":
            print(f'For Rain200L dataset PSNR: {qm_psnr:.4f} SSIM: {qm_ssim:.4f}')
        else:
            print(f'For {dataset} dataset PSNR: {qm_psnr:.4f} SSIM: {qm_ssim:.4f}')
        return qm_psnr, qm_ssim
    return 0, 0
