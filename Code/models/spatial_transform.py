import torch
import torch.nn.functional as F
import math

'''
Reference:
Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in neural information processing systems 28 (2015).
'''


def corner2matrix(params):
    '''
    Convert two corner displacements to transformation matrix.

    Args:
        params: tensor of batch*4

    Returns:
        transformation matrices of batch*2*3
    '''
    shape_r = (128, 128)
    x_bias = int(shape_r[0] / 2)
    y_bias = int(shape_r[1] / 2)
    c1 = [0 - x_bias, 0 - y_bias]
    c2 = [0 - x_bias, shape_r[1] - y_bias]
    c_o = [c1, c2]
    c_o = torch.tensor(c_o, dtype=torch.float32).reshape(4, 1).cuda()

    final_r = torch.zeros((params.shape[0], 2, 3)).cuda()
    for i in range(params.shape[0]):
        c_t = c_o + params[i, :].view(4, 1)
        l_m = [[c_o[0], -c_o[1], 1, 0], [c_o[1], c_o[0], 0, 1], [c_o[2], -c_o[3], 1, 0], [c_o[3], c_o[2], 0, 1]]
        l_m = torch.tensor(l_m, dtype=torch.float32).cuda()
        vec = torch.matmul(torch.inverse(l_m), c_t)
        r1 = torch.index_select(vec, 0, torch.tensor([0, 1, 2]).cuda()).view(1, 3).cuda()
        r1 = r1.mul(torch.tensor([1, 1, 1 / shape_r[0]]).view(1, 3).cuda())
        r2 = torch.index_select(vec, 0, torch.tensor([1, 0, 3]).cuda()).view(1, 3).cuda()
        r2 = r2.mul(torch.tensor([-1, 1, 1 / shape_r[1]]).view(1, 3).cuda())
        tm = torch.cat((r1, r2), dim=0)
        final_r[i, :] = tm

    return final_r


def matrix2torch(m):
    '''
    Convert transformation matrix to PyTorch standard affine matrix.

    Args:
        m: transformation matrices of batch*2*3

    Returns:
        transformation matrices of batch*2*3
    '''
    tm = torch.zeros(m.size()).cuda()
    for i in range(m.shape[0]):
        slices = torch.zeros((2, 3), dtype=torch.float32).cuda()
        slices[0, 0] = m[i, 0, 0]
        slices[0, 1] = -m[i, 0, 1]
        slices[1, 0] = -m[i, 1, 0]
        slices[1, 1] = m[i, 1, 1]
        slices[0, 2] = -m[i, 1, 2] * 2
        slices[1, 2] = -m[i, 0, 2] * 2
        tm[i, :] = slices
    return tm


def spatial_transform(x, params, padding='zeros'):
    '''
    Apply spatial transformation to images in format corner displacements.

    Args:
        x: input images of size batch*h*w
        params: two corner displacements of size batch*4
        padding: padding mode of spatial transformation which is predefined by PyTorch

    Returns:
        x: transformed images
        matrix_back: transformation matrices of params
        mask: mask generated by transformation
    '''
    matrix = corner2matrix(params)
    matrix_back = torch.zeros((params.shape[0], 3, 3)).cuda()
    matrix_back[:, 0:2, 0:3] = matrix[:, 0:2, 0:3]
    matrix_back[:, 2, 2] = 1
    matrix_r = torch.zeros(matrix.shape).cuda()
    for i in range(matrix.shape[0]):
        matrix_r[i, 0:2, 0:2] = matrix[i, 0:2, 0:2]
    matrix_r = matrix2torch(matrix_r)
    grid = F.affine_grid(matrix_r, x.size())
    x = F.grid_sample(x, grid, padding_mode=padding)
    mask = torch.ones(x.shape).cuda()
    mask = F.grid_sample(mask, grid, padding_mode=padding)
    mask[mask != 0] = 1

    matrix_s = torch.zeros(matrix.shape).cuda()
    for i in range(matrix.shape[0]):
        matrix_s[i, 0, 0] = 1
        matrix_s[i, 1, 1] = 1
        matrix_s[i, :, 2] = matrix[i, :, 2]
    matrix_s = matrix2torch(matrix_s)
    grid = F.affine_grid(matrix_s, x.size())
    x = F.grid_sample(x, grid, padding_mode=padding)
    mask = F.grid_sample(mask, grid, padding_mode=padding)
    mask[mask != 0] = 1
    return x, matrix_back, mask


def spatial_transform_angle(x, params, padding='zeros'):
    '''
    Apply spatial transformation to images in format rotaion angles.

    Args:
        x: input images of size batch*h*w
        params: rotaion angles of size batch*1
        padding: padding mode of spatial transformation which is predefined by PyTorch

    Returns:
        x: transformed images
        matrix_back: transformation matrices of params
        mask: mask generated by transformation
    '''
    matrix = torch.zeros((params.shape[0], 2, 3)).cuda()
    for i in range(matrix.shape[0]):
        matrix[i, 0, 0] = torch.cos(params[i] / 360 * 2 * math.pi)
        matrix[i, 0, 1] = torch.sin(params[i] / 360 * 2 * math.pi)
        matrix[i, 1, 0] = -torch.sin(params[i] / 360 * 2 * math.pi)
        matrix[i, 1, 1] = torch.cos(params[i] / 360 * 2 * math.pi)
    matrix_back = torch.zeros((params.shape[0], 3, 3)).cuda()
    matrix_back[:, 0:2, 0:2] = matrix[:, 0:2, 0:2]
    matrix_back[:, 2, 2] = 1
    matrix = matrix2torch(matrix)
    grid = F.affine_grid(matrix, x.size())
    x = F.grid_sample(x, grid, padding_mode=padding)
    mask = torch.ones(x.shape).cuda()
    mask = F.grid_sample(mask, grid, padding_mode=padding)
    mask[mask != 0] = 1
    return x, matrix_back, mask


def spatial_transform_nomask(x, params, padding='reflection'):
    '''
    Apply spatial transformation to images in format corner displacements. No mask is generated.

    Args:
        x: input images of size batch*h*w
        params: two corner displacements of size batch*4
        padding: padding mode of spatial transformation which is predefined by PyTorch

    Returns:
        transformed images
    '''
    matrix = corner2matrix(params)
    matrix_r = torch.zeros(matrix.shape).cuda()
    for i in range(matrix.shape[0]):
        matrix_r[i, 0:2, 0:2] = matrix[i, 0:2, 0:2]
    matrix_r = matrix2torch(matrix_r)
    grid = F.affine_grid(matrix_r, x.size())
    x = F.grid_sample(x, grid, padding_mode=padding)

    matrix_s = torch.zeros(matrix.shape).cuda()
    for i in range(matrix.shape[0]):
        matrix_s[i, 0, 0] = 1
        matrix_s[i, 1, 1] = 1
        matrix_s[i, :, 2] = matrix[i, :, 2]
    matrix_s = matrix2torch(matrix_s)
    grid = F.affine_grid(matrix_s, x.size())
    x = F.grid_sample(x, grid, padding_mode=padding)
    return x


def spatial_transform_angle_nomask(x, params, padding='reflection'):
    '''
    Apply spatial transformation to images in format rotaion angles. No mask is generated.

    Args:
        x: input images of size batch*h*w
        params: rotaion angles of size batch*1
        padding: padding mode of spatial transformation which is predefined by PyTorch

    Returns:
        transformed images
    '''
    matrix = torch.zeros((params.shape[0], 2, 3)).cuda()
    for i in range(matrix.shape[0]):
        matrix[i, 0, 0] = torch.cos(params[i] / 360 * 2 * math.pi)
        matrix[i, 0, 1] = torch.sin(params[i] / 360 * 2 * math.pi)
        matrix[i, 1, 0] = -torch.sin(params[i] / 360 * 2 * math.pi)
        matrix[i, 1, 1] = torch.cos(params[i] / 360 * 2 * math.pi)
    matrix = matrix2torch(matrix)
    grid = F.affine_grid(matrix, x.size())
    x = F.grid_sample(x, grid, padding_mode=padding)
    return x
