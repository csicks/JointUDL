import glob
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils import read_img, range_one, normalize, img2fourier


def get_data(path):
    '''
    Get all image names in one folder.

    Args:
        path: path to images

    Returns:
        image names in list
    '''
    files = glob.glob(path + "*.png")
    files.sort()
    return files


class RealData(Dataset):
    '''
    Class for real-world cryo-EM dataset.
    '''
    def __init__(self, data, norm):
        '''
        Initialization.

        Args:
            data: list containing data
            norm: whether to apply normalization to images
        '''
        super(RealData, self).__init__()
        self.images = data
        self.norm = norm

    def __len__(self):
        '''
        Returns:
            length of dataset
        '''
        return len(self.images)

    def read(self, idx):
        '''
        Read image from disk according to its index in dataset.

        Args:
            idx: index of image in dataset (the dataset is sorted in alphabetical order)

        Returns:
            image in ndarray
        '''
        name = self.images[idx]
        image = read_img(name)
        image = range_one(image)
        return image

    def __getitem__(self, idx):
        pass


class RealReal(RealData):
    def __getitem__(self, idx):
        '''
        Get image according to its index in dataset.

        Args:
            idx: index of image in dataset (the dataset is sorted in alphabetical order)

        Returns:
            image list (each image in ndarray)
        '''
        image = self.read(idx)

        idx0 = idx - 1 if idx > 0 else self.__len__() - 1
        image0 = self.read(idx0)
        if self.norm:
            image_r = np.array([normalize(image0), normalize(image)], dtype=np.float32)
        else:
            image_r = np.array([image0, image], dtype=np.float32)
        return image_r


class RealFourier(RealData):
    def __getitem__(self, idx):
        '''
        Get image's Fourier spectrum according to its index in dataset.

        Args:
            idx: index of image in dataset (the dataset is sorted in alphabetical order)

        Returns:
            image_r: image's Fourier spectrum list (each spectrum in ndarray)
            image list (each image in ndarray)
        '''
        image = self.read(idx)
        f = img2fourier(image)
        f = range_one(f)

        idx0 = idx - 1 if idx > 0 else self.__len__() - 1
        image0 = self.read(idx0)
        f0 = img2fourier(image0)
        f0 = range_one(f0)
        if self.norm:
            image_r = np.array([normalize(f0), normalize(f)], dtype=np.float32)
        else:
            image_r = np.array([f0, f], dtype=np.float32)
        return image_r, np.array([image0, image], dtype=np.float32)


def real_real(path, batch, norm=False):
    '''
    Get real-world cryo-EM dataset in spatial domain.

    Args:
        path: path to dataset
        batch: batch size
        norm: whether to apply normalization to images

    Returns:
        dataloader for training
    '''
    images = get_data(path)
    random.shuffle(images)
    dataset = RealReal(images, norm)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4)
    return data_loader


def real_fourier(path, batch, norm=False):
    '''
    Get real-world cryo-EM dataset in Fourier domain.

    Args:
        path: path to dataset
        batch: batch size
        norm: whether to apply normalization to images

    Returns:
        dataloader for training
    '''
    images = get_data(path)
    random.shuffle(images)
    dataset = RealFourier(images, norm)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4)
    return data_loader
