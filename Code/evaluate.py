import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import us_synthetic_fourier, us_joint_fourier
from models import spatial_transform_angle_nomask, angle_loss, init_kaiming_normal
from models.resnet import ResidualBlockDeep
from utils import Logger, img2fourier, range_one
import numpy as np
import time


class Net(nn.Module):
    '''
    Class for the overall architecture.
    '''
    def __init__(self, ):
        super(Net, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.feature = nn.Sequential(
            ResidualBlockDeep(1,
                              32,
                              stride=1,
                              expansion=1,
                              shortcut=nn.Sequential(
                                  nn.Conv2d(1, 32, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(32))),  # 64
            nn.MaxPool2d(2),
            ResidualBlockDeep(32,
                              64,
                              stride=1,
                              expansion=1,
                              shortcut=nn.Sequential(
                                  nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(64))),  # 32
            nn.MaxPool2d(2),
            ResidualBlockDeep(64,
                              128,
                              stride=1,
                              expansion=1,
                              shortcut=nn.Sequential(
                                  nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(128))),  # 16
            nn.MaxPool2d(2),
            ResidualBlockDeep(128,
                              256,
                              stride=1,
                              expansion=1,
                              shortcut=nn.Sequential(
                                  nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(256))),  # 8
            nn.MaxPool2d(2),
            ResidualBlockDeep(256,
                              512,
                              stride=1,
                              expansion=1,
                              shortcut=nn.Sequential(
                                  nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(512))),  # 4
            nn.MaxPool2d(2),
            ResidualBlockDeep(512,
                              1024,
                              stride=1,
                              expansion=1,
                              shortcut=nn.Sequential(
                                  nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(1024))),  # 2
            nn.MaxPool2d(2),
            ResidualBlockDeep(1024,
                              2048,
                              stride=1,
                              expansion=1,
                              shortcut=nn.Sequential(
                                  nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                                  nn.BatchNorm2d(2048))),  # 1
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(2048 * 2, 2000), nn.Linear(2000, 2000),
                                nn.Linear(2000, 1, bias=False))

    def forward(self, x):
        '''
        Forward function.

        Args:
            x: input image pairs of size batch*2*128*128

        Returns:
            rotation angles of size batch*1
        '''
        x0 = torch.unsqueeze(x[:, 0, :, :], 1)
        x1 = torch.unsqueeze(x[:, 1, :, :], 1)
        m0 = self.mask(x0)
        m1 = self.mask(x1)
        f0 = self.feature(x0 * m0)
        f1 = self.feature(x1 * m1)
        r0 = f0.reshape(f0.size(0), -1)
        r1 = f1.reshape(f1.size(0), -1)
        r = torch.cat((r0, r1), dim=1)
        xs = self.fc(r)
        return xs


def diff2bias_angle(output, target):
    output = output % 180
    target = target % 180
    r = ((target - output) % 360) % 180
    b = 0
    for i in range(output.shape[0]):
        temp = r[i]
        if temp > 150:
            temp_s = temp - 180
        else:
            temp_s = temp + 180
        if torch.abs(temp) < torch.abs(temp_s):
            b += temp
        else:
            b += temp_s
    b /= output.shape[0]
    return b


def evaluate(model, train_loader, device, logger):
    '''
    Function to train the network.

    Args:
        epoch: epoch number
        model: network
        train_loader: dataloader for training
        device: cpu or gpu
        optimizer: optimizer
        scheduler: scheduler
        logger: logger to record traning progress
    '''
    bias = 0 ### estimation
    with torch.no_grad():
        model.train()
        s = 0
        count = 0
        for batch_idx, (data, target, image) in enumerate(train_loader):
            data, target, image = data.to(device), target.to(device), image.to(device)
            output0 = model(data)
            target_loss = angle_loss((output0 + bias) % 360, target).item()
            s += target_loss
            count += 1
            logger.write(str(target_loss) + '\n')
        s = s / count
        logger.write('Average Error: %f\n' % (s))
        logger.write('Bias : %f\n' % (bias))


def main_synthetic(logger, checkpoint_path):
    '''
    Main function. Note in cryo-EM single particle images, we aim to align this dataset accurately without groundtruth instead of training a generalizable model (which may also be possible). Therefore, only training function is given.

    Args:
        logger: logger to record training/testing progress
        name: name for checkpoint
        continue_flag: train from the begining or continue training from a checkpoint
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 32
    path = '/path/to/synthetic data'
    index = [5]
    train_loader = us_joint_fourier(path, batch, fc=False, fn=True, picked=index)

    logger.write("Datasets: " + str(index) + '\n')

    model = Net().to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    st = time.time()
    evaluate(model, train_loader, device, logger)
    et = time.time()
    logger.write('Time: %f second' % (et - st))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    checkpoint_path = '/path/to/check point'
    logger = Logger('hist.txt', print_label=False)
    main_synthetic(logger, checkpoint_path)
        
