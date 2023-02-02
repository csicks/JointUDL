import glob
import math
import os
import random
import numpy as np
from utils import params2matrix, params2corner, read_img, range_one, normalize, img2fourier
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


def get_single(data_path, folder, func):
    '''
    Get both images and groundtruth transformation parameters from one folder.

    Args:
        data_path: path to folder containing data
        func: function to deal with transformation parameters (convert it to certain format)

    Returns:
        r_data: image name pairs in list
        r_txt: groundtruth transformation parameters in list
    '''
    data_path = data_path + folder + '/'
    f_data = get_data(data_path + 'one_png/')
    f_txt = get_txt(data_path + 'parameters.txt')

    r_data = []
    r_txt = []
    for i in range(len(f_data)):
        if i != len(f_data) - 1:
            r_data.append([f_data[i], f_data[i + 1]])
            angle0, x0, y0 = f_txt[i]
            angle1, x1, y1 = f_txt[i + 1]
        else:
            r_data.append([f_data[i], f_data[0]])
            angle0, x0, y0 = f_txt[i]
            angle1, x1, y1 = f_txt[0]
        m0 = params2matrix(angle0, x0, y0)
        m1 = params2matrix(angle1, x1, y1)
        mn = np.dot(m0, np.linalg.inv(m1))
        angle_n = math.atan2(mn[0, 1], mn[0, 0]) / 2 / np.pi * 360
        angle_n = angle_n + 360 if angle_n < 0 else angle_n
        x_n = mn[1, 2]
        y_n = mn[0, 2]
        r_txt.append(func(angle_n, x_n, y_n))

    return r_data, r_txt


class OneData(Dataset):
    '''
    Class for synthetic cryo-EM dataset in one folder.
    '''
    def __init__(self, data, norm):
        super(OneData, self).__init__()
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
        '''
        Get image's Fourier spectrum according to its index in dataset.

        Args:
            idx: index of image in dataset (the dataset is sorted in alphabetical order)

        Returns:
            image_r: image's Fourier spectrum list (each spectrum in ndarray)
            label_r: groundtruth transformation parameters list
            image list (each image in ndarray)
        '''
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


def one_fourier(path, folder, batch, norm=False, shuffle=True):
    '''
    Get synthetic cryo-EM dataset in Fourier domain for images in one folder. For UDL.

    Args:
        path: path to dataset
        folder: folder name
        norm: whether to apply normalization to images
        shuffle: whether to shuffle dataset

    Returns:
        dataloader for training
    '''
    data_all = get_single(path, folder, lambda a, x, y: [a])
    images, labels = data_all
    rand_num = random.randint(0, 100)
    random.seed(rand_num)
    random.shuffle(images)
    random.seed(rand_num)
    random.shuffle(labels)

    data_train = [images, labels]
    dataset_train = OneData(data_train, norm)
    train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=shuffle, num_workers=4)

    return train_loader
