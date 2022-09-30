import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from datasets import eval_fourier
from models import spatial_transform_angle_nomask
from models.resnet import ResidualBlockDeep
from utils import Logger
import numpy as np


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
            nn.MaxPool2d(2),
            ResidualBlockDeep(256, 512, stride=1, expansion=1, shortcut=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512)
            )),  # 4
            nn.MaxPool2d(2),
            ResidualBlockDeep(512, 1024, stride=1, expansion=1, shortcut=nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(1024)
            )),  # 2
            nn.MaxPool2d(2),
            ResidualBlockDeep(1024, 2048, stride=1, expansion=1, shortcut=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048)
            )),  # 1
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048 * 2, 2000),
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
        r0 = f0.reshape(f0.size(0), -1)
        r1 = f1.reshape(f1.size(0), -1)
        r = torch.cat((r0, r1), dim=1)
        xs = self.fc(r)
        return xs


def correntropy(img1, img2):
    '''
    Calculate the correntropy between two images.

    Args:
        img1: the first image in ndarray
        img2: the second image in ndarray

    Returns:
        correntropy value
    '''
    s = np.mean(np.exp(-np.power(img1 - img2, 2)))
    return s


def generate(model, train_loader, device, logger):
    '''
    Generate txt file and average image.

    Args:
        model: well-trained model
        train_loader: data_loader
        device: cpu or gpu
        logger: logger to record the progress
    '''
    s = np.zeros((128, 128), dtype=np.float32)
    sim = 0
    count = 0
    model.eval()
    bias = 0
    angles = []
    with torch.no_grad():
        for batch_idx, (data, image) in enumerate(train_loader):
            data, image = data.to(device), image.to(device)

            output0 = model(data)
            a = output0.detach().cpu().numpy()[0, 0]
            angles.append(a)

            image0 = torch.unsqueeze(image[:, 0, :, :], dim=1)
            image1 = torch.unsqueeze(image[:, 1, :, :], dim=1)
            image2 = spatial_transform_angle_nomask(image1, output0 + bias)

            image0s = torch.squeeze(image0[0, :, :, :]).detach().cpu().numpy()
            image2s = torch.squeeze(image2[0, :, :, :]).detach().cpu().numpy()

            s += image2s
            corr = correntropy(image0s, image2s)
            sim += corr
            count += 1
            logger.write(str(a) + '\n')
    s /= count
    sim /= count
    plt.imshow(s, plt.cm.gray)
    plt.show()
    plt.imsave('./average.png', s)


def main_synthetic(logger):
    '''
    Main function.

    Args:
        logger: logger to record the progress
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 1
    path = '/path/to/image folder'
    checkpoint_path = '/path/to/well-trained model'
    train_loader = eval_fourier(path, batch, '/path/to/reference image')

    model = Net().to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    generate(model, train_loader, device, logger)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    name = os.path.basename(__file__).split('.')[0:-1][0]

    logger = Logger(name + '.txt')
    main_synthetic(logger)
