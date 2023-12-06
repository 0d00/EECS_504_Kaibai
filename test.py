import datetime
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
import os
import time
from unet_dataset import UnetDataset
from unet import UNet
import matplotlib.pyplot as plt
import glob
import tqdm
from celluloid import Camera
from IPython.display import HTML
import ffmpeg
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load('./model/unet_best.pt'))
    model.eval()
    test_dataset = UnetDataset('test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, num_workers=4, shuffle=False)
    fig = plt.figure(figsize=(10, 10))
    camera = Camera(fig)
    for x, y in tqdm.tqdm(test_dataset):
        input = torch.tensor([x]).to(device, dtype=torch.float32)
        predict = model(input)
        predict=predict.detach().cpu().numpy()
        mask_predict = (predict[0][0] > 0.5)

        plt.subplot(1, 2, 1)
        plt.imshow(x[0], cmap='bone')
        mask_img1 = np.ma.masked_where(y[0] == 0, y[0])
        plt.imshow(mask_img1, alpha=0.8, cmap="spring")
        plt.title('truth')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(x[0], cmap='bone')
        mask_img2 = np.ma.masked_where(mask_predict== 0, mask_predict)
        plt.imshow(mask_img2, alpha=0.8, cmap="spring")
        plt.title('prediction')
        plt.axis('off')
        camera.snap()


    animation = camera.animate()
    animation.save("demo.mp4")