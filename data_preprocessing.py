import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tqdm
import glob
import os
import cv2

data = pd.read_csv("data//metadata.csv")


def replace_file_path(path):
    path = path.replace("../input/covid19-ct-scans", "data")
    return path


def read_nii(filename):
    img = nib.load(filename)
    img = img.get_fdata()
    img = np.rot90(np.array(img))
    return img


image = [replace_file_path(path) for path in data.iloc[:, 0]]
label = [replace_file_path(path) for path in data.iloc[:, 2]]

for i, f in tqdm.tqdm(enumerate(image)):
    data_len = len(image)
    train_len = int(0.7 * data_len)
    dir = ""
    if i < train_len:
        dir = "data/train/"
    else:
        dir = "data/test/"
    img = nib.load(f)
    mask = nib.load(label[i])
    img_num = img.get_fdata()
    mask_num = mask.get_fdata().astype(np.uint8)
    mean = np.mean(img_num)
    std = np.std(img_num)
    img_num = (img_num - mean) / std
    img_max = img_num.max()
    img_min = img_num.min()
    img_num = (img_num - img_min) / (img_max - img_min)
    layer_num = img_num.shape[-1]
    for j in range(layer_num):
        layer = img_num[:, :, j]
        layer_mask = mask_num[:, :, j]
        layer = cv2.resize(layer, (256, 256))
        layer_mask = cv2.resize(layer_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        path = dir + str(i)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "/img_" + str(j), layer)
        np.save(path + "/label_" + str(j), layer_mask)

