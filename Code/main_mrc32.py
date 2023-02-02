import os
import torch
import torch.nn as nn
from models import ResidualBlockDeep, spatial_transform_angle_nomask
from utils import Logger, range_one, img2fourier, normalize
import numpy as np
import mrcfile
from torch.utils.data import Dataset, DataLoader
import cv2
import argparse
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


class EMData(Dataset):
    '''
    Class for cryo-EM data in MRCS files.
    '''
    def __init__(self, data, ref, norm):
        super(EMData, self).__init__()
        self.images = data
        self.ref = ref
        if self.ref.shape[0] != 128 or self.ref.shape[1] != 128:
            self.ref = cv2.resize(self.ref, (128, 128))
        self.ref = range_one(self.ref)
        self.ref_f = img2fourier(self.ref)
        self.ref_f = range_one(self.ref_f)
        self.norm = norm

    def __len__(self):
        return self.images.shape[0]

    def read(self, idx):
        image = self.images[idx, :]
        image_ori = image.copy()
        if image.shape[0] != 128 or image.shape[1] != 128:
            image = cv2.resize(image, (128, 128))
        image = range_one(image)
        return image, image_ori

    def __getitem__(self, idx):
        image, image_ori = self.read(idx)
        f = img2fourier(image)
        f = range_one(f)

        if self.norm:
            image_r = np.array([normalize(self.ref_f), normalize(f)], dtype=np.float32)
        else:
            image_r = np.array([self.ref_f, f], dtype=np.float32)
        return image_r, image_ori


def get_data(path):
    '''
    Read data from MRCS file path.

    Args:
        path: path to MRCS file

    Returns:
        image data in ndarray
    '''
    mrc = mrcfile.open(path)
    data = mrc.data
    return data


def write_data(img, path):
    '''
    Write data into MRCS file path.

    Args:
        img: data in ndarray
        path: path to MRCS file
    '''
    if os.path.exists(path):
        os.remove(path)
    new_mrc = mrcfile.new(path)
    new_mrc.set_data(img)


def em_fourier(data, batch, ref, norm=False):
    '''
    Get cryo-EM dataset for joint models.

    Args:
        path: path to dataset
        batch: batch size
        ref: reference image ndarray
        norm: whether to apply normalization to images

    Returns:
        dataloader for models
    '''
    images = data
    dataset = EMData(images, ref, norm)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    return data_loader


def correntropy(img1, img2):
    '''
    Correntropy for image correlation.

    Args:
        img1: the first image in ndarray
        img2: the second image in ndarray

    Returns:
        correntropy value
    '''
    s = np.mean(np.exp(-np.power(img1 - img2, 2)))
    return s


def read_xmd(path):
    '''
    Read data from XMIPP xmd files (only for certain formats)

    Args:
        path: path to XMIPP xmd file

    Returns:
        data: image data in ndarray
        info: header information of XMIPP xmd file
    '''
    file = open(path, 'r')
    lines = file.readlines()
    info = []

    mrc = ['', None]
    data = []

    for i in range(len(lines)):
        if i < 5:
            continue
        line = lines[i]
        txt_group = line.split('@')
        if len(txt_group) != 2:
            break
        info.append(line)

        if txt_group[1] != mrc[0]:
            mrc[1] = get_data(mrc[0])
        data.append(mrc[1][int(txt_group[0])])

    data = np.array(data, dtype=np.float32)
    return data, info


def generate_info(data, path):
    '''
    Generate header for XMIPP xmd files (only for certain formats)

    Args:
        data: image data in ndarray
        path: path to XMIPP xmd file

    Returns:
        header information of XMIPP xmd file
    '''
    info = []
    length = len(str(data.shape[0]))
    for i in range(data.shape[0]):
        info.append(str(i).zfill(length) + '@' + path)
    return info


def main(logger, path, model_path, out_name):
    '''
    Main function. Read data from MRCS/XMD file and apply alignment. Information about transformation parameters and aligned image stack are generated.

    Args:
        logger: logger to record training/testing progress
        path: path to MRCS/XMD file
        model_path: path to well-trained model in PKL file
        out_name: name for outputs
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 32

    temp = path.split('.')
    ext = temp[len(temp) - 1]

    if ext == 'mrcs':
        data = get_data(path)
        info = generate_info(data, path)
    elif ext == 'xmd':
        data, info = read_xmd(path)
    else:
        print('Wrong input!')
        raise

    ref = data[0]

    model = Net().to(device)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    data_loader = em_fourier(data, batch, ref)

    model.train()

    bias = 0 ## estimated
    image_all = []
    out_all = []
    with torch.no_grad():
        for batch_idx, (data, image) in enumerate(data_loader):
            data, image = data.to(device), image.to(device)
            output = model(data)
            image = torch.unsqueeze(image, dim=1)
            image_trans = spatial_transform_angle_nomask(image, output + bias)
            image_trans = torch.squeeze(image_trans).detach().cpu().numpy()
            for i in range(image_trans.shape[0]):
                image_all.append(image_trans[i,:,:])
                out_all.append(output.detach().cpu().numpy()[i, 0])

    file = open(out_name + '_alignment.xmd', 'a')
    file.write("# XMIPP_STAR_1 * \n")
    file.write("# \n")
    file.write("data_noname")
    file.write("loop_\n")
    file.write(" _image\n")
    file.write(" _shiftX\n")
    file.write(" _shiftY\n")
    file.write(" _anglePsi\n")
    file.write(" _flip\n")
    file.write(" _maxCC\n")

    for i in range(len(image_all)):
        image = image_all[i]
        cc = correntropy(ref, image)
        message = info[i] + '\t0\t0\t' + str(out_all[i]) + '\t0\t' + str(cc) + '\n'
        file.write(message)
    file.close()

    image_all = np.array(image_all, dtype=np.float32)
    write_data(image_all, out_name + '.mrcs')


if __name__ == '__main__':
    logger = Logger('mrc.txt')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    st = time.time()
    main(logger, '/path/to/MRCS file', 
         '/path/to/check point', 'name')
    et = time.time()
    print('Time cost %f\n' % (et - st))

