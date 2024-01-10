import argparse
import os
from glob import glob

import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from scipy.io import savemat, loadmat
from tqdm import tqdm

"""
该段代码用于生成训练数据集
"""
parser = argparse.ArgumentParser(description='Generate patches from Full Resolution mat files')
parser.add_argument('--src_dir', default='../datasets/WDC/', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/WDC/train', type=str, help='Directory for image patches')
parser.add_argument('--ps', default=64, type=int, help='Image Patch Size')  # 单个样本的大小 64x64
parser.add_argument('--num_patches', default=1024, type=int, help='Number of patches per image')  # 样本数量
parser.add_argument('--num_cores', default=8, type=int, help='Number of CPU Cores')
parser.add_argument('--train_flag', action='store_true', default=True)
parser.add_argument('--K', default=24, type=int, help='Number of bands')  # 频段数

args = parser.parse_args()

NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores
ADJ_BANDS = args.K

src = args.src_dir
tar = args.tar_dir
patch_size = args.ps
train_flag = args.train_flag

# 检查target路径是否存在, 不存在则创建路径
if not os.path.exists(tar):
    os.makedirs(tar)

# natsorted用于对字符串进行自然排序, glob用于匹配特定模式的文件路径名
files = natsorted(glob(src + '/*.mat'))
print(files)


# 存储文件
def save_output(path, data, name):
    data_dict = {"gt": data}
    savemat(path + "/" + name + ".mat", data_dict)


def save_files(i, key):
    clean_file = files[i]
    clean_img = loadmat(clean_file)

    height = clean_img[key].shape[1]
    width = clean_img[key].shape[2]
    bands = clean_img[key].shape[0]

    # 沿波段拓展
    clean = np.concatenate([clean_img[key][range(int(ADJ_BANDS / 2), 0, -1), ...],
                            clean_img[key],
                            clean_img[key][range(bands - 1, bands - int(ADJ_BANDS / 2), -1), ...]], axis=0)

    for j in range(NUM_PATCHES):
        h = np.random.randint(0, height - patch_size)
        w = np.random.randint(0, width - patch_size)
        b = np.random.randint(0, bands)

        clean_patch = clean[b:b + ADJ_BANDS, h:h + patch_size, w:w + patch_size]

        save_output(tar, clean_patch, 'patch_{}_{}'.format(i + 1, j + 1))


# Parallel用于并行计算
# tqdm用于在cmd中添加进度条
# delayed函数用于将需要并行计算的函数和参数包装, 便于Parallel并行计算
Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i, 'image') for i in tqdm(range(0, 1)))
