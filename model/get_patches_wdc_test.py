import argparse
import os
from glob import glob

import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from scipy.io import savemat, loadmat
from tqdm import tqdm

"""
该段代码用于生成测试数据集
"""
parser = argparse.ArgumentParser(description='Generate patches from Full Resolution mat files')
parser.add_argument('--src_dir', default='../datasets/WDC/test', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/WDC/test/blind', type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=1, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=8, type=int, help='Number of CPU Cores')
parser.add_argument('--K', default=24, type=int, help='Number of bands')

args = parser.parse_args()

NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores
ADJ_BANDS = args.K

src = args.src_dir
tar = args.tar_dir
PS = args.ps

if not os.path.exists(tar):
    os.makedirs(tar)

files = natsorted(glob(os.path.join(src, '*.mat')))
print(files)

clean_file = files[0]
print(clean_file)
clean_img = loadmat(clean_file)
bands = clean_img['gt'].shape[0]
clean = np.concatenate([clean_img['gt'][range(int(ADJ_BANDS / 2), 0, -1), ...],
                        clean_img['gt'],
                        clean_img['gt'][range(bands - 1, bands - int(ADJ_BANDS / 2), -1), ...]], axis=0)
noisy = np.concatenate([clean_img['input'][range(int(ADJ_BANDS / 2), 0, -1), ...],
                        clean_img['input'],
                        clean_img['input'][range(bands - 1, bands - int(ADJ_BANDS / 2), -1), ...]], axis=0)


def save_output(path, gt, decrease, name):
    data_dict = {"gt": gt, "input": decrease}
    savemat(path + "/" + name + ".mat", data_dict)


def save_files(i):
    for j in range(NUM_PATCHES):
        for k in range(0, bands, 1):
            clean_patch = clean[k:k + int(ADJ_BANDS), i * PS:i * PS + PS, j * PS:j * PS + PS]
            noisy_patch = noisy[k:k + int(ADJ_BANDS), i * PS:i * PS + PS, j * PS:j * PS + PS]

            save_output(tar, clean_patch, noisy_patch, 'patch_{}_{}_{}'.format(i + 1, j + 1, k + 1))


Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(NUM_PATCHES)))
