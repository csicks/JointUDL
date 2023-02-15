import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AngleLoss(nn.Module):
    '''
    Class for loss function of cryo-EM single particle images where particles are usually centrosymmetric. For centrosymmetric particles, rotation angle is a periodic signal of 180 degree.
    '''
    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, r, t):
        '''
        Forward function.

        Args:
            r: result of network output (not guranteed to be in range [0,360])
            t: groundtruth (guranteed to be in range [0,360])

        Returns:
            loss value
        '''
        s = 0
        ts = t.clone()
        ts[t < 180] += 180
        ts[t >= 180] -= 180
        for i in range(r.shape[0]):
            s += torch.min(F.l1_loss(r[i], t[i]), F.l1_loss(r[i], ts[i]))
        return s / r.shape[0]


def angle_loss(result, target):
    '''
    same as AngleLoss class defined before
    '''
    f = AngleLoss()
    return f(result, target)


class PeriodLoss(nn.Module):
    '''
    Class for loss function of nature images. For nature images, rotation angle is a periodic signal of 360 degree.
    '''
    def __init__(self):
        super(PeriodLoss, self).__init__()

    def forward(self, r, t):
        '''
        Forward function.

        Args:
            r: result of network output (not guranteed to be in range [0,360])
            t: groundtruth (guranteed to be in range [0,360])

        Returns:
            loss value
        '''
        r = r % 360
        t = t % 360
        s = 0
        ts = t.clone()
        ts[t < 180] += 360
        ts[t >= 180] -= 360
        for i in range(r.shape[0]):
            s += torch.min(F.l1_loss(r[i], t[i]), F.l1_loss(r[i], ts[i]))
        return s / r.shape[0]


def period_loss(result, target):
    '''
    same as PeriodLoss class defined before
    '''
    f = PeriodLoss()
    return f(result, target)
