import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset, DataLoader


class UnetDataset(Dataset):
    def __init__(self,stage):
        self.data = glob.glob('data/{}/*/img_*'.format(stage))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file_path = self.data[index]
        mask_file_path = img_file_path.replace('img', 'label')
        img = np.load(img_file_path)
        mask = np.load(mask_file_path)
        return np.expand_dims(img, 0), np.expand_dims(mask, 0)



