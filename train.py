import sys

import torch
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
import os
import time


from unet_dataset import UnetDataset
from unet import UNet
import datetime


def config(parser):
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)


    args = parser.parse_args()
    return args


def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./log/',
                only_file=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                            )


class trainer():
    def __init__(self, args, model, train_data, test_data):
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_works = args.num_works
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = args.epochs
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                        num_workers=self.num_works, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size,
                                                       num_workers=self.num_works, shuffle=False)

    def get_test_loss(self, loader, model):
        loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device, dtype=torch.float32)

                y = y.to(self.device, dtype=torch.float32)

                predict = model(x)

                loss_batch = self.loss_fn(predict, y)
                loss += loss_batch
        return loss / len(loader)

    def train(self):
        cnt = 20
        best_test_loss = 1e12
        for epoch in range(self.epochs):
            self.model.train()
            start_time = time.time()
            loss = 0
            for i, (x, y) in enumerate(tqdm.tqdm(self.train_loader)):
                self.model.zero_grad()
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                predict = self.model(x)
                loss_batch = self.loss_fn(predict, y)
                loss_batch.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_batch = loss_batch.detach().cpu()
                loss += loss_batch

            loss = loss / len(self.train_loader)
            test_loss = self.get_test_loss(self.test_loader, self.model)
            cnt -= 1
            if best_test_loss > test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), './model/unet_best.pt')
                logging.info('EPOCH{} get lowest loss:{}'.format(epoch, best_test_loss))
                cnt = 20
            logging.info('epoch{}: time:{}s，train loss：{}，test loss：{}\n'.format(
                epoch,
                time.time() - start_time,
                loss,
                test_loss
            ))
            if cnt == 0:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = config(parser)
    logger_init(log_level=logging.INFO)
    logging.info(args)
    train_dataset = UnetDataset('train')
    test_dataset = UnetDataset('test')
    model = UNet()

    trainer = trainer(args, model, train_dataset, test_dataset)
    trainer.train()
