import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import s_coco_real_angle
from models import init_kaiming_normal, period_loss, spatial_transform_nomask
from models.resnet import ResidualBlockDeep
from utils import Logger, params2corner


class MatchNet(nn.Module):
    '''
    Class for feature matching layer.

    Reference: Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
    '''
    def __init__(self, norm=True):
        '''
        Initialization.
        
        Args:
            whether to apply feature normalization
        '''
        super(MatchNet, self).__init__()
        self.norm = norm
        self.relu = nn.ReLU(True)

    def feature_norm(self, feature):
        '''
        Feature normalization.
        '''
        feature = self.relu(feature)
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

    def forward(self, f0, f1):
        '''
        Forward function.

        Args:
            f0: the first input feature of size batch*channel*h*w
            f1: the second input feature of size batch*channel*h*w

        Returns:
            feature correlation map of size batch*(h*w)*h*w
        '''
        b, c, h, w = f0.size()
        f0_c = f0.view(b, c, h * w)
        f1_c = f1.view(b, c, h * w).transpose(1, 2)
        f_mul = torch.bmm(f1_c, f0_c)
        corr = f_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        if self.norm:
            corr = self.feature_norm(corr)
        return corr


class Net(nn.Module):
    '''
    Class for the overall architecture.
    '''
    def __init__(self, ):
        super(Net, self).__init__()
        self.match = MatchNet()
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
            ResidualBlockDeep(1, 32, stride=1, expansion=1, shortcut=nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(32)
            )),  # 64
            nn.MaxPool2d(2),
            ResidualBlockDeep(32, 64, stride=1, expansion=1, shortcut=nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64)
            )),  # 32
            nn.MaxPool2d(2),
            ResidualBlockDeep(64, 128, stride=1, expansion=1, shortcut=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128)
            )),  # 16
            nn.MaxPool2d(2),
            ResidualBlockDeep(128, 256, stride=1, expansion=1, shortcut=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256)
            )),  # 8
            nn.MaxPool2d(2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8 * 2, 2000),
            nn.Linear(2000, 2000),
            nn.Linear(2000, 1, bias=False)
        )

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
        c0 = self.match(f0, f1)
        c1 = self.match(f1, f0)
        r0 = self.conv(c0)
        r1 = self.conv(c1)
        r0 = r0.reshape(r0.size(0), -1)
        r1 = r1.reshape(r1.size(0), -1)
        r = torch.cat((r0, r1), dim=1)
        xs = self.fc(r)
        return xs


def random_transform(data):
    '''
    Apply manual disturbance (rotation and translations) to input image pair to generate another image pair.

    Args:
        data: input image pairs of size batch*2*128*128

    Returns:
        data_t: disturbed image pairs
        dc: groundtruth disturbance parameters (difference value between two rotation angle)
    '''
    data0 = torch.unsqueeze(data[:, 0, :, :], dim=1)
    data1 = torch.unsqueeze(data[:, 1, :, :], dim=1)
    dc0 = torch.zeros((data.shape[0], 1), dtype=torch.float32).cuda()
    dc1 = torch.zeros((data.shape[0], 1), dtype=torch.float32).cuda()
    cdc0 = torch.zeros((data.shape[0], 4), dtype=torch.float32).cuda()
    cdc1 = torch.zeros((data.shape[0], 4), dtype=torch.float32).cuda()
    for i in range(data.shape[0]):
        da = random.uniform(0, 360)
        dx = random.uniform(-5, 5)
        dy = random.uniform(-5, 5)
        tdc = params2corner(da, dx, dy)
        tdc = torch.tensor(tdc, dtype=torch.float32)
        cdc0[i, :] = tdc
        dc0[i, :] = da
        da = random.uniform(0, 360)
        dx = random.uniform(-5, 5)
        dy = random.uniform(-5, 5)
        tdc = params2corner(da, dx, dy)
        tdc = torch.tensor(tdc, dtype=torch.float32)
        cdc1[i, :] = tdc
        dc1[i, :] = da
    data0_t = spatial_transform_nomask(data0, cdc0)
    data1_t = spatial_transform_nomask(data1, cdc1)
    data_t = torch.cat((data0_t, data1_t), dim=1)
    dc = (dc0 - dc1) % 360
    return data_t, dc


def diff2bias(output, target):
    '''
    Function to calculate bias for each batch of data. Groundtruth value is only used to generate bias and evaluate the performance of each training epoch but does not participate in training. The calculated bias which is a single float value is given for testing, which could be estimated from any number of input image pairs in practice (with such number of groundtruth required).

    Args:
        output: network outputs of size batch*1
        target: groundtruth value

    Returns:
        bias for this batch of data
    '''
    r = (target - output) % 360
    b = 0
    for i in range(output.shape[0]):
        temp = r[i]
        if temp > 300:
            temp_s = temp - 360
        else:
            temp_s = temp + 360
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
        logger: logger to record training progress

    Returns:
        bias for the whole dataset
    '''
    model.train()
    total_loss = 0
    target_loss = 0
    biasR = torch.zeros(len(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output0 = model(data)
        loss_r0 = F.margin_ranking_loss(output0, torch.zeros_like(output0), torch.ones_like(output0)) + \
                  F.margin_ranking_loss(output0, 360 * torch.ones_like(output0), -torch.ones_like(output0))

        data_t, dm = random_transform(data)

        output1 = model(data_t)
        loss_r1 = F.margin_ranking_loss(output1, torch.zeros_like(output1), torch.ones_like(output1)) + \
                  F.margin_ranking_loss(output1, 360 * torch.ones_like(output1), -torch.ones_like(output1))

        mid_m = (output1 - output0) % 360
        loss = period_loss(mid_m, dm) + loss_r0 + loss_r1

        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()

        bias = diff2bias(output0, target).item()
        target_loss += period_loss((output0 + bias) % 360, target).item()
        biasR[batch_idx] = bias

    bias_std, bias_mean = torch.std_mean(biasR)
    logger.write('Train Epoch: {} \tLoss: {:.6f}\tTarget Loss: {:.6f}\tBias Mean: {:.6f}\tBias Std: {:.6f}\n'.format(
        epoch,
        total_loss / (len(train_loader.dataset) / train_loader.batch_size),
        target_loss / (len(train_loader.dataset) / train_loader.batch_size),
        bias_mean,
        bias_std
    ))
    return bias_mean.item()


def test(epoch, model, test_loader, device, logger, bias):
    '''
    Function to teat the network.

    Args:
        epoch: epoch number
        model: network
        test_loader: dataloader for testing
        device: cpu or gpu
        logger: logger to record testing progress
        bias: bias value estimated from training
    '''
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = period_loss((output + bias) % 360, target)
            test_loss += loss.item()

        test_loss /= len(test_loader.dataset) / test_loader.batch_size
        logger.write('Test Epoch: {} \tLoss: {:.6f}\n'
                     .format(epoch, test_loss))


def main_synthetic(logger, name, continue_flag=False):
    '''
    Main function.

    Args:
        logger: logger to record training/testing progress
        name: name for checkpoint
        continue_flag: train from the begining or continue training from a checkpoint
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 64
    path = '/path/to/synthetic MS-COCO dataset/'
    train_loader, test_loader = s_coco_real_angle(path, batch, label=['sp'], norm=True)
    checkpoint_path = name + '.pkl'

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12000, gamma=0.8)

    if continue_flag:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0
        init_kaiming_normal(model)
        torch.save({'model_state_dict': model.state_dict()}, name + '_init.pkl')

    for epoch in range(current_epoch + 1, 100 + 1):
        bias = train(epoch, model, train_loader, device, optimizer, scheduler, logger)
        test(epoch, model, test_loader, device, logger, bias)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, checkpoint_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    name = os.path.basename(__file__).split('.')[0:-1][0]
    mode = 'train'

    if mode == 'train':
        logger = Logger(name + '.txt')
        main_synthetic(logger, name, continue_flag=False)
