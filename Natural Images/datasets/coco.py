import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import params2corner, read_img, range_one, normalize
import random
import cv2


def get_data_c(path):
    '''
    Get image names of name "*_c.png" in one folder.

    Args:
        path: path to images

    Returns:
        image names in list
    '''
    files = glob.glob(path + "*_c.png")
    files.sort()
    return files


def get_data_d(path):
    '''
    Get image names of name "*_d.png" in one folder.

    Args:
        path: path to images

    Returns:
        image names in list
    '''
    files = glob.glob(path + "*_d.png")
    files.sort()
    return files


def get_txt(path):
    '''
    Get groundtruth transformation parameters between "*_c.png" and "*_d.png".

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
    Get both images and groundtruth transformation parameters from one folder.

    Args:
        data_path: path to folder containing data
        func: function to deal with transformation parameters (convert it to certain format)

    Returns:
        r_data: image name pairs in list
        r_txt: groundtruth transformation parameters in list
    '''
    f_data_c = get_data_c(data_path + 'one_png/')
    f_data_d = get_data_d(data_path + 'one_png/')
    f_txt = get_txt(data_path + 'parameters.txt')

    r_data = []
    r_txt = []
    for i in range(len(f_txt)):
        r_data.append([f_data_d[i], f_data_c[i]])
        angle, x, y = f_txt[i]
        r_txt.append(func(angle, x, y))

    return r_data, r_txt


def get_all(data_path, func, label):
    '''
    Get both images and groundtruth transformation parameters from the whole dataset.

    Args:
        data_path: path to dataset
        func: function to deal with transformation parameters (convert it to certain format)

    Returns:
        data_all: all image name pairs in list
        target_all: all groundtruth transformation parameters in list
    '''
    data_all = []
    target_all = []

    f_list = []

    if 'clean' in label:
        f_list.append(data_path + 'clean/')
    if 'gaussian' in label:
        f_list.append(data_path + 'gaussian/')
    if 'sp' in label:
        f_list.append(data_path + 'sp/')

    for f_path in f_list:
        r_data, r_txt = get_single(f_path, func)
        data_all.extend(r_data)
        target_all.extend(r_txt)

    return data_all, target_all


class COCO(Dataset):
    '''
    Class for synthetic MS-COCO dataset.
    '''
    def __init__(self, data, norm):
        '''
        Initialization.

        Args:
            data: list containing data and labels in format [data,labels]
            norm: whether to apply normalization to images
        '''
        super(COCO, self).__init__()
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
        Get image according to its index in dataset.

        Args:
            idx: index of image in dataset (the dataset is sorted in alphabetical order)

        Returns:
            image_r: image list (each image in ndarray)
            label_r: groundtruth transformation parameters list
        '''
        image0, image1, label = self.read(idx)
        if self.norm:
            image_r = np.array([normalize(image0), normalize(image1)], dtype=np.float32)
        else:
            image_r = np.array([image0, image1], dtype=np.float32)
        label_r = np.array(label, dtype=np.float32)
        return image_r, label_r


def s_coco_real(path, batch, label, norm=True):
    '''
    Get synthetic MD-COCO dataset using two corner displacements as groundtruth label.

    Args:
        path: path to dataset
        batch: batch size
        label: which dataset to get (clean, Gaussian noisy or salt-and-pepper noisy)
        norm: whether to apply normalization to images

    Returns:
        train_loader: dataloader for training
        val_loader: dataloader for validation/testing
    '''
    path_train = path + 'train/'
    data_train = get_all(path_train, params2corner, label)

    path_val = path + 'val/'
    data_val = get_all(path_val, params2corner, label)

    dataset_train = COCO(data_train, norm=norm)
    dataset_val = COCO(data_val, norm=norm)

    train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def s_coco_real_angle(path, batch, label, norm=True):
    '''
    Get synthetic MD-COCO dataset using rotation angle as groundtruth label.

    Args:
        path: path to dataset
        batch: batch size
        label: which dataset to get (clean, Gaussian noisy or salt-and-pepper noisy)
        norm: whether to apply normalization to images

    Returns:
        train_loader: dataloader for training
        val_loader: dataloader for validation/testing
    '''
    path_train = path + 'train/'
    data_train = get_all(path_train, lambda a, x, y: [a], label)

    path_val = path + 'val/'
    data_val = get_all(path_val, lambda a, x, y: [a], label)

    dataset_train = COCO(data_train, norm=norm)
    dataset_val = COCO(data_val, norm=norm)

    train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
