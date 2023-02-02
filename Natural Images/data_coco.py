import cv2
import numpy as np
import os
import shutil
import random
import glob


def shift(img, x, y):
    '''
    Shift image.

    Args:
        img: image in ndarray
        x: tranlation pixels in x axis (down direction as positive direction)
        y: tranlation pixels in y axis (right direction as positive direction)

    Returns:
        translated image in ndarray
    '''
    if x == 0 and y == 0:
        return img
    rows, cols = img.shape
    matrix = np.float32([[1, 0, y], [0, 1, x]])
    img = cv2.warpAffine(img, matrix, (cols, rows), borderMode=cv2.BORDER_WRAP)
    return img


def rotate(img, angle, center):
    '''
    Rotate image.

    Args:
        img: image in ndarray
        angle: rotation angle in degree (counterclockwise as positive direction)
        center: coordinate of rotation center

    Returns:
        rotated image in ndarray
    '''
    rows, cols = img.shape
    matrix = cv2.getRotationMatrix2D((center[1], center[0]), angle, 1)
    img = cv2.warpAffine(img, matrix, (cols, rows), borderMode=cv2.BORDER_WRAP)
    return img


def gaussian_noise(img, var=-1):
    '''
    Add Gaussian noise to image.

    Args:
        img: image in ndarray
        var: variaration rate of Gaussian noise added to image, and SNR would be 1/var

    Returns:
        Gaussian noisy image in ndarray
    '''
    mean = 0
    if var == -1:
        var = np.var(img) * 10
    else:
        var = np.var(img) * var
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    return out


def ps_noise(img, p0=0.2):
    '''
    Add salt-and-pepper noise to image.

    Args:
        img: image in ndarray
        p0: propotion of salt/pepper noise in image

    Returns:
        salt-and-pepper noisy image in ndarray
    '''
    p1 = 1 - p0
    out = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = random.random()
            if r < p0:
                out[i, j] = 0
            elif r > p1:
                out[i, j] = 255
    return out


def to_array(img):
    '''
    Adjust image range to [0,255]

    Args:
        img: image in ndarray

    Returns:
        adjusted image in ndarray
    '''
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.array(img * 255, dtype=np.uint8)
    return img


def process(path_in, path_out):
    '''
    Generate synthetic dataset.

    Args:
        path_in: path to input images (in one folder)
        path_out: path to generated dataset
    '''
    files = glob.glob(path_in + "*.jpg")
    length = len(str(len(files)))
    files.sort()
    s_shape = (128, 128)
    rx = int(s_shape[0] / 2)
    ry = int(s_shape[1] / 2)
    shift_max = 10
    gaussian_rate = 10 # variaration rate of Gaussian noise added to image, and SNR would be 1/gaussian_rate
    sp_rate = 0.2 #propotion of salt/pepper noise in image

    out_path_clean = path_out + "clean/"
    os.mkdir(out_path_clean)
    one_path_clean = out_path_clean + "one_png/"
    os.mkdir(one_path_clean)
    txt_path_clean = out_path_clean + "parameters.txt"
    f_clean = open(txt_path_clean, 'w')

    out_path_gaussian = path_out + "gaussian/"
    os.mkdir(out_path_gaussian)
    one_path_gaussian = out_path_gaussian + "one_png/"
    os.mkdir(one_path_gaussian)
    txt_path_gaussian = out_path_gaussian + "parameters.txt"
    f_gaussian = open(txt_path_gaussian, 'w')

    out_path_sp = path_out + "sp/"
    os.mkdir(out_path_sp)
    one_path_sp = out_path_sp + "one_png/"
    os.mkdir(one_path_sp)
    txt_path_sp = out_path_sp + "parameters.txt"
    f_sp = open(txt_path_sp, 'w')

    count = 0
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if rx + shift_max >= img.shape[0] - shift_max - rx or ry + shift_max >= img.shape[1] - shift_max - ry:
            continue
        angle = random.random() * 360
        shift_x = random.randint(-shift_max, shift_max)
        shift_y = random.randint(-shift_max, shift_max)
        cx = random.randint(rx + shift_max, img.shape[0] - shift_max - rx)
        cy = random.randint(ry + shift_max, img.shape[1] - shift_max - ry)
        img_d = shift(rotate(img, angle, (cx, cy)), shift_x, shift_y)
        patch = img[cx - rx:cx + rx, cy - ry:cy + ry]
        patch_d = img_d[cx - rx:cx + rx, cy - ry:cy + ry]
        if np.var(patch) == 0 or np.var(patch_d) == 0:
            continue
        path_cc = one_path_clean + str(count).zfill(length) + '_c.png'
        path_cd = one_path_clean + str(count).zfill(length) + '_d.png'
        cv2.imwrite(path_cc, to_array(patch))
        cv2.imwrite(path_cd, to_array(patch_d))

        if os.path.getsize(path_cc) < 6 * 1024 or os.path.getsize(path_cd) < 6 * 1024:
            os.remove(path_cc)
            os.remove(path_cd)
            continue

        patch_g = gaussian_noise(patch, gaussian_rate)
        patch_gd = gaussian_noise(patch_d, gaussian_rate)
        cv2.imwrite(one_path_gaussian + str(count).zfill(length) + '_c.png', to_array(patch_g))
        cv2.imwrite(one_path_gaussian + str(count).zfill(length) + '_d.png', to_array(patch_gd))

        patch_sp = ps_noise(patch, sp_rate)
        patch_spd = ps_noise(patch_d, sp_rate)
        cv2.imwrite(one_path_sp + str(count).zfill(length) + '_c.png', to_array(patch_sp))
        cv2.imwrite(one_path_sp + str(count).zfill(length) + '_d.png', to_array(patch_spd))

        f_clean.write(str(count) + ' ' + str(angle) + ' ' + str(shift_x) + ' ' + str(shift_y) + '\n')
        f_gaussian.write(str(count) + ' ' + str(angle) + ' ' + str(shift_x) + ' ' + str(shift_y) + '\n')
        f_sp.write(str(count) + ' ' + str(angle) + ' ' + str(shift_x) + ' ' + str(shift_y) + '\n')

        count += 1
    f_clean.close()
    f_gaussian.close()
    f_sp.close()


def train(path_in, path_out):
    '''
    Generate synthetic MS-COCO dataset for training.

    Args:
        path_in: path to input images (in one folder)
        path_out: path to generated dataset
    '''
    path_train_in = path_in + 'train2017/'
    path_train_out = path_out + 'train/'
    os.mkdir(path_train_out)
    process(path_train_in, path_train_out)


def val(path_in, path_out):
    '''
    Generate synthetic MS-COCO dataset for validation/testing.

    Args:
        path_in: path to input images (in one folder)
        path_out: path to generated dataset
    '''
    path_val_in = path_in + 'val2017/'
    path_val_out = path_out + 'val/'
    os.mkdir(path_val_out)
    process(path_val_in, path_val_out)


if __name__ == '__main__':
    path_in = '/path/to/COCO/' # like folder 'MS-COCO samples' in Supplementary File
    path_out = '/path/to/output/'
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.mkdir(path_out)
    train(path_in, path_out)
    val(path_in, path_out)
