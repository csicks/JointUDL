import cv2
import numpy as np



def read_img(path):
    '''
    Read image and convert it to grayscale.

    Args:
        path: path to image

    Returns:
        image in ndarray
    '''
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.array(img, dtype=np.float32)
    return img


def img2fourier(img, log=True):
    '''
    Convert image to its Fourier spectrum.

    Args:
        img: image in ndarray
        log: whether to apply logarithm function to Fourier spectrum.

    Returns:
        Fourier spectrum of image (log=False) or logarithm of Fourier spectrum (log=True)
    '''
    f = np.fft.fft2(img)
    f = np.fft.fftshift(f)
    f = np.abs(f)
    if log:
        f = np.log(f + 1)
    return f


def range_one(img):
    '''
    Convert image to [0,1] range.

    Args:
        img: image in ndarray

    Returns:
        image in ndarray in range [0,1]
    '''
    img = np.array(img, dtype=np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def normalize(img):
    '''
    Apply normalization to image

    Args:
        img: image in ndarray

    Returns:
        normalizated image in ndarray
    '''
    m = np.mean(img)
    img = (img - m) / np.std(img)
    return img


def params2matrix(angle, x, y, img_size=(128, 128)):
    '''
    Convert rigid transformation parameters rotation angle and two translations (horizontal and vertical) to
    transformation matrix of size 3*3

    Args:
        angle: rotation angle in degree
        x: vertical translation in pixel
        y: horizontal translation in pixel
        img_size: size of image

    Returns:
        transformation matrix of size 3*3
    '''
    m = np.zeros((3, 3))
    cols, rows = img_size
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    matrix[0, 2] = y
    matrix[1, 2] = x
    m[0, 0], m[0, 1], m[0, 2] = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m[1, 0], m[1, 1], m[1, 2] = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m[2, 0], m[2, 1], m[2, 2] = 0, 0, 1
    return m


def params2corner(angle, dx, dy):
    '''
    Convert rigid transformation parameters rotation angle and two translations (horizontal and vertical) to corner displacements (left top corner and right top corner). The origin of the coordinate system is the center point of image.

    Args:
        angle: rotation angle in degree
        dx: vertical translation in pixel
        dy: horizontal translation in pixel

    Returns:
        list of four elements (x displacement of left top corner, y displacement of left top corner, x displacement of right top corner, y displacement of right top corner)
    '''
    angle = -angle / 360 * 2 * np.pi
    shape_r = [128, 128]
    x_bias = int(shape_r[0] / 2)
    y_bias = int(shape_r[1] / 2)
    c1 = [0 - x_bias, 0 - y_bias]
    c2 = [0 - x_bias, shape_r[1] - y_bias]
    c_o = [c1, c2]
    r = []
    for each in c_o:
        xn = each[0] * np.cos(angle) + each[1] * np.sin(angle) + dx
        yn = -each[0] * np.sin(angle) + each[1] * np.cos(angle) + dy
        r.append(xn - each[0])
        r.append(yn - each[1])
    return r
