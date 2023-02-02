import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import us_synthetic_fourier
from models import spatial_transform_angle_nomask, angle_loss, init_kaiming_normal
from models.resnet import ResidualBlockDeep
from utils import Logger, img2fourier, range_one
import numpy as np


class Net(nn.Module):
    '''
    Class for the overall architecture. 
    For cryo-EM image datasets where all images are of the same class,feature matching is not very 
    necessary and does not improve the performance significantly. Therefore,feature matching is removed 
    here and you may refer to 'Natural Images' part for code and implementation.
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


def random_transform(data):
    '''
    Apply manual disturbance (only rotation, since translations are omitted in Fourier spectrum) to input image pair to generate another image pair.

    Args:
        data: input image pairs of size batch*2*128*128

    Returns:
        data_t: Fourier spectrums of disturbed image pairs
        dc: groundtruth disturbance parameters (difference value between two rotation angle)
    '''
    data0 = torch.unsqueeze(data[:, 0, :, :], dim=1)
    data1 = torch.unsqueeze(data[:, 1, :, :], dim=1)
    dc0 = torch.zeros((data.shape[0], 1), dtype=torch.float32).cuda()
    dc1 = torch.zeros((data.shape[0], 1), dtype=torch.float32).cuda()
    for i in range(data.shape[0]):
        da = random.uniform(0, 360)
        dc0[i, :] = da
        da = random.uniform(0, 360)
        dc1[i, :] = da
    data0_t = spatial_transform_angle_nomask(data0, dc0)
    data0_t = torch.squeeze(data0_t).detach().cpu().numpy()
    data1_t = spatial_transform_angle_nomask(data1, dc1)
    data1_t = torch.squeeze(data1_t).detach().cpu().numpy()
    f0 = np.zeros(data0_t.shape, dtype=np.float32)
    f1 = np.zeros(data1_t.shape, dtype=np.float32)
    for i in range(data0_t.shape[0]):
        ft0 = img2fourier(data0_t[i, :])
        ft0 = range_one(ft0)
        f0[i, :] = ft0
        ft1 = img2fourier(data1_t[i, :])
        ft1 = range_one(ft1)
        f1[i, :] = ft1
    f0 = torch.from_numpy(f0)
    f1 = torch.from_numpy(f1)
    f0 = torch.unsqueeze(f0, dim=1).cuda()
    f1 = torch.unsqueeze(f1, dim=1).cuda()
    data_t = torch.cat((f0, f1), dim=1)
    dc = (dc0 - dc1) % 360
    return data_t, dc


def diff2bias_angle(output, target):
    '''
    Function to calculate bias for each batch of data. Groundtruth value is only used to generate bias and evaluate the performance of each training epoch but does not participate in training. Note the rotation angle is of period 180 degree here (centrosymmetric).

    Args:
        output: network outputs of size batch*1
        target: groundtruth value

    Returns:
        bias for this batch of data
    '''
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


def train(epoch, model, train_loader, device, optimizer, scheduler, logger):
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
    model.train()
    total_loss = 0
    target_loss = 0
    biasR = torch.zeros(len(train_loader))
    for batch_idx, (data, target, image) in enumerate(train_loader):
        data, target, image = data.to(device), target.to(device), image.to(device)

        optimizer.zero_grad()
        output0 = model(data)
        loss_r0 = F.margin_ranking_loss(output0, torch.zeros_like(output0), torch.ones_like(output0)) + \
                  F.margin_ranking_loss(output0, 360 * torch.ones_like(output0), -torch.ones_like(output0))

        data_t, dm = random_transform(image)
        output1 = model(data_t)
        loss_r1 = F.margin_ranking_loss(output1, torch.zeros_like(output1), torch.ones_like(output1)) + \
                  F.margin_ranking_loss(output1, 360 * torch.ones_like(output1), -torch.ones_like(output1))

        mid_m = (output1 - output0) % 360

        loss = angle_loss(mid_m, dm) + loss_r0 + loss_r1

        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()
        bias = diff2bias_angle(output0, target).item()
        target_loss += angle_loss((output0 + bias) % 360, target).item()
        biasR[batch_idx] = bias

    bias_std, bias_mean = torch.std_mean(biasR)
    logger.write(
        'Train Epoch: {} \tLoss: {:.6f}\tTarget Loss: {:.6f}\tBias Mean: {:.6f}\tBias Std: {:.6f}\n'.
        format(epoch, total_loss / (len(train_loader.dataset) / train_loader.batch_size),
               target_loss / (len(train_loader.dataset) / train_loader.batch_size), bias_mean, bias_std))


def main_synthetic(logger, name, continue_flag=False):
    '''
    Main function. Note in cryo-EM single particle images, we aim to align this dataset accurately without groundtruth instead of training a generalizable model (which may also be possible). Therefore, only training function is given.

    Args:
        logger: logger to record training/testing progress
        name: name for checkpoint
        continue_flag: train from the beginning or continue training from a checkpoint
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 32
    path = '/path/to/synthetic data'
    index = [4]
    train_loader = us_synthetic_fourier(path, batch, fc=False, fn=True, picked=index)
    checkpoint_path = name + '.pkl'

    logger.write("Datasets: " + str(index) + '\n')

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12000, gamma=0.8)

    if continue_flag:
        checkpoint = torch.load('/path/to/check point')
        model.load_state_dict(checkpoint['model_state_dict'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0
        init_kaiming_normal(model)

    for epoch in range(current_epoch + 1, current_epoch + 101):
        train(epoch, model, train_loader, device, optimizer, scheduler, logger)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, checkpoint_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    name = os.path.basename(__file__).split('.')[0:-1][0]
    mode = 'train'

    if mode == 'train':
        logger = Logger(name + '.txt')
        main_synthetic(logger, name, continue_flag=True)
