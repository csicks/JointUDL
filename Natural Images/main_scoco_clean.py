import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import s_coco_real_angle
from models import init_kaiming_normal, period_loss
from models.resnet import ResidualBlockDeep
from utils.logger import Logger


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
    '''
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.smooth_l1_loss(output, target)

        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()

    logger.write('Train Epoch: {} \tLoss: {:.6f}\n'.format(
        epoch, total_loss / (len(train_loader.dataset) / train_loader.batch_size)
    ))


def test(epoch, model, test_loader, device, logger):
    '''
    Function to test the network.

    Args:
        epoch: epoch number
        model: network
        test_loader: dataloader for testing
        device: cpu or gpu
        logger: logger to record testing progress
    '''
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = period_loss(output, target)
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
    train_loader, test_loader = s_coco_real_angle(path, batch, label=['clean'], norm=True)
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
        train(epoch, model, train_loader, device, optimizer, scheduler, logger)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, checkpoint_path)
        test(epoch, model, test_loader, device, logger)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    name = os.path.basename(__file__).split('.')[0:-1][0]
    mode = 'train'

    if mode == 'train':
        logger = Logger(name + '.txt')
        main_synthetic(logger, name, continue_flag=False)
