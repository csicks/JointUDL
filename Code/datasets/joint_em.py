import glob
import math
import os
import random
import numpy as np
from utils import params2matrix, read_img, range_one, normalize, img2fourier
from torch.utils.data import Dataset, DataLoader


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


def get_txt(path):
    '''
    Get groundtruth transformation parameters between one image and the image following it (i.e. the i_th image and the i+1_th image and the last image is followed by the first image).

    Args:
        path: path to labeling file such as "parameters.txt"

    Returns:
        transformation parameters of one rotaion angle and two translations in list
    '''
    f = open(path, 'r')
    s = f.read()
    f.close()
    s_group = []
    for each in s.split('\n'):
        data = each.split(' ')
        if len(data) <= 1:
            break
        s_group.append([float(data[1]), float(data[2]), float(data[3])])
    return s_group


def get_single(data_path, func):
    '''
    Get both images and groundtruth transformation parameters from one folder. Note every image in dataset is combined with the first image in dataset to create image pairs for joint alignment.

    Args:
        data_path: path to folder containing data
        func: function to deal with transformation parameters (convert it to certain format)

    Returns:
        r_data: image name pairs in list
        r_txt: groundtruth transformation parameters in list
    '''
    f_data = get_data(data_path + 'one_png/')
    f_txt = get_txt(data_path + 'parameters.txt')

    r_data = []
    r_txt = []
    for i in range(len(f_data)):
        if i != 0:
            r_data.append([f_data[0], f_data[i]])
            angle0, x0, y0 = f_txt[0]
            angle1, x1, y1 = f_txt[i]
        else:
            continue
        m0 = params2matrix(angle0, x0, y0)
        m1 = params2matrix(angle1, x1, y1)
        mn = np.dot(m0, np.linalg.inv(m1))
        angle_n = math.atan2(mn[0, 1], mn[0, 0]) / 2 / np.pi * 360
        angle_n = angle_n + 360 if angle_n < 0 else angle_n
        x_n = mn[1, 2]
        y_n = mn[0, 2]
        r_txt.append(func(angle_n, x_n, y_n))

    return r_data, r_txt


def get_all(data_path, func, fc, fn, picked=None):
    '''
    Get both images and groundtruth transformation parameters from the whole dataset.

    Args:
        data_path: path to dataset
        func: function to deal with transformation parameters (convert it to certain format)
        fc: whether to use clean dataset
        fc: whether to use noisy dataset
        picked: which dataset to use (list containing integers)

    Returns:
        data_all: all image name pairs in list
        target_all: all groundtruth transformation parameters in list
    '''
    if picked is None:
        picked = [0]
    folders = os.listdir(data_path)
    folders.sort()
    data_all = []
    target_all = []

    folder_list = []
    for number in picked:
        folder_list.append(folders[number])

    for folder in folder_list:
        f_path = data_path + folder + '/'
        f_list = []

        if fc:
            f_list.append(f_path + 'clean/')
        if fn:
            f_list.append(f_path + 'noise/')

        for f_path in f_list:
            r_data, r_txt = get_single(f_path, func)
            data_all.extend(r_data)
            target_all.extend(r_txt)

    return data_all, target_all


class SyntheticData(Dataset):
    '''
    Class for synthetic cryo-EM dataset.
    '''

    def __init__(self, data, norm):
        super(SyntheticData, self).__init__()
        self.images, self.labels = data
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
            image0: the first image in ndarray
            image1: the second image in ndarray
            label: groundtruth transformation parameters
        '''
        pair = self.images[idx]
        image0 = read_img(pair[0])
        image1 = read_img(pair[1])
        image0 = range_one(image0)
        image1 = range_one(image1)

        label = self.labels[idx]
        return image0, image1, label

    def __getitem__(self, idx):
        pass


class JointReal(SyntheticData):
    '''
    Get image according to its index in dataset for joint alignment.

    Args:
        idx: index of image in dataset (the dataset is sorted in alphabetical order)

    Returns:
        image_r: image list (each image in ndarray)
        label_r: groundtruth transformation parameters list
    '''

    def __getitem__(self, idx):
        image0, image1, label = self.read(idx)
        if self.norm:
            image_r = np.array([normalize(image0), normalize(image1)], dtype=np.float32)
        else:
            image_r = np.array([image0, image1], dtype=np.float32)
        label_r = np.array(label, dtype=np.float32)
        return image_r, label_r


class JointFourier(SyntheticData):
    '''
    Get image's Fourier spectrum according to its index in dataset for joint alignment.

    Args:
        idx: index of image in dataset (the dataset is sorted in alphabetical order)

    Returns:
        image_r: image's Fourier spectrum list (each spectrum in ndarray)
        label_r: groundtruth transformation parameters list
        image list (each image in ndarray)
    '''

    def __getitem__(self, idx):
        image0, image1, label = self.read(idx)
        f0 = img2fourier(image0)
        f1 = img2fourier(image1)
        f0 = range_one(f0)
        f1 = range_one(f1)
        if self.norm:
            image_r = np.array([normalize(f0), normalize(f1)], dtype=np.float32)
        else:
            image_r = np.array([f0, f1], dtype=np.float32)
        label_r = np.array(label, dtype=np.float32)
        return image_r, label_r, np.array([image0, image1], dtype=np.float32)


def joint_base(data_all, synthetic_dataset, batch, norm=False, shuffle=True):
    '''
    Base function for getting synthetic cryo-EM dataset for joint alignment.

    Args:
        data_all: list containing data and labels
        synthetic_dataset: dataset class to use
        batch: batch size
        norm: whether to apply normalization to images
        shuffle: whether to shuffle dataset

    Returns:
        dataloader for training
    '''
    images, labels = data_all
    rand_num = random.randint(0, 100)
    random.seed(rand_num)
    random.shuffle(images)
    random.seed(rand_num)
    random.shuffle(labels)

    data_train = [images, labels]
    dataset_train = synthetic_dataset(data_train, norm)
    train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=shuffle, num_workers=4)
    return train_loader


def us_joint_real(path, batch, fc, fn, picked=None, norm=False):
    '''
    Get synthetic cryo-EM dataset in spatial domain for joint alignment.

    Args:
        path: path to dataset
        batch: batch size
        fc: whether to use clean dataset
        fc: whether to use noisy dataset
        picked: which dataset to use (list containing integers)
        norm: whether to apply normalization to images

    Returns:
        dataloader for training
    '''
    if picked is None:
        picked = [0]
    data_all = get_all(path, lambda a, x, y: [a], fc=fc, fn=fn, picked=picked)
    train_loader = joint_base(data_all, JointReal, batch, norm=norm)
    return train_loader


def us_joint_fourier(path, batch, fc, fn, picked=None, norm=False):
    '''
    Get synthetic cryo-EM dataset in Fourier domain for joint alignment.

    Args:
        path: path to dataset
        batch: batch size
        fc: whether to use clean dataset
        fc: whether to use noisy dataset
        picked: which dataset to use (list containing integers)
        norm: whether to apply normalization to images

    Returns:
        dataloader for training
    '''
    if picked is None:
        picked = [0]
    data_all = get_all(path, lambda a, x, y: [a], fc=fc, fn=fn, picked=picked)
    train_loader = joint_base(data_all, JointFourier, batch, norm=norm)
    return train_loader
