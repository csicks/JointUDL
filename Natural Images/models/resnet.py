import torch.nn as nn
import torch.nn.functional as F

'''
Reference:
He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
'''


class ResidualBlockShallow(nn.Module):
    '''
    Class for residual block used in ResNet18 and ResNet34.
    '''
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        '''
        Initialization.

        Args:
            in_channel: channel of input tensor
            out_channel: channel of output tensor
            stride: stride of the first convolutional layer in residual block
            shortcut: layers to feed input tensor directly to output tensor; in implementation, it consists of one convolutional layer of kernel size 1*1 and one batch normalization layer
        '''
        super(ResidualBlockShallow, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        '''
        Forward function.

        Args:
            x: input tensor of size batch*in_channel*h*w

        Returns:
            output feature of size batch*out_channel*(h/stride)*(w/stride)
        '''
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNetShallow(nn.Module):
    '''
    Class for ResNet18 and ResNet34.
    '''
    def __init__(self, blocks, num_classes=4):
        '''
        Initialization.

        Args:
            blocks: a list containing number of layers of each stage
            num_classes: number of outputs of the last fully connected layer
        '''
        super(ResNetShallow, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(2, 64, 7, 2, 3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self.make_layer(64, 128, blocks[0])
        self.layer2 = self.make_layer(128, 256, blocks[1], stride=2)
        self.layer3 = self.make_layer(256, 512, blocks[2], stride=2)
        self.layer4 = self.make_layer(512, 512, blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes, bias=False)

    def make_layer(self, in_channel, out_channel, block_num, stride=1):
        '''
        Generate layers according to given parameters.

        Args:
            in_channel: channel of input tensor
            out_channel: channel of output tensor
            block_num: total number of residual blocks
            stride: stride of the first residual block

        Returns:
            generated layers
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = [ResidualBlockShallow(in_channel, out_channel, stride, shortcut)]
        for i in range(1, block_num):
            layers.append(ResidualBlockShallow(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward function.

        Args:
            x: input tensor of size batch*2*128*128

        Returns:
            output parameters of size batch*num_classes
        '''
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlockDeep(nn.Module):
    '''
    Class for residual block used in ResNet50, ResNet101 and ResNet152.
    '''
    def __init__(self, in_channel, out_channel, stride=1, expansion=4, shortcut=None):
        '''
        Initialization.

        Args:
            in_channel: channel of input tensor
            out_channel: channel of output tensor
            stride: stride of the first convolutional layer in residual block
            expansion: expansion rate of channel of the last convolutional layer
            shortcut: layers to feed input tensor directly to output tensor; in implementation, it consists of one convolutional layer of kernel size 1*1 and one batch normalization layer
        '''
        super(ResidualBlockDeep, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel * expansion)
        )
        self.right = shortcut

    def forward(self, x):
        '''
        Forward function.

        Args:
            x: input tensor of size batch*in_channel*h*w

        Returns:
            output feature of size batch*(out_channel*expansion)*(h/stride)*(w/stride)
        '''
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNetDeep(nn.Module):
    '''
    Class for ResNet50, ResNet101 and ResNet152.
    '''
    def __init__(self, blocks, num_classes=4):
        super(ResNetDeep, self).__init__()
        self.expansion = 4
        self.pre = nn.Sequential(
            nn.Conv2d(2, 64, 7, 2, 3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self.make_layer(64, 64, blocks[0])
        self.layer2 = self.make_layer(256, 128, blocks[1], stride=2)
        self.layer3 = self.make_layer(512, 256, blocks[2], stride=2)
        self.layer4 = self.make_layer(1024, 512, blocks[3], stride=2)
        self.fc = nn.Linear(2048, num_classes, bias=False)

    def make_layer(self, in_channel, out_channel, block_num, stride=1):
        '''
        Generate layers according to given parameters.

        Args:
            in_channel: channel of input tensor
            out_channel: channel of output tensor
            block_num: total number of residual blocks
            stride: stride of the first residual block

        Returns:
            generated layers
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * self.expansion, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion)
        )
        layers = [ResidualBlockDeep(in_channel, out_channel, stride, self.expansion, shortcut)]
        for i in range(1, block_num):
            layers.append(ResidualBlockDeep(out_channel * self.expansion, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward function.

        Args:
            x: input tensor of size batch*2*128*128

        Returns:
            output parameters of size batch*num_classes
        '''
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18(n_classes=4):
    '''
    ResNet18.

    Args:
        n_classes: number of output parameters

    Returns:
        network ResNet18
    '''
    return ResNetShallow([2, 2, 2, 2], n_classes)


def resnet34(n_classes=4):
    '''
    ResNet34.

    Args:
        n_classes: number of output parameters

    Returns:
        network ResNet34
    '''
    return ResNetShallow([3, 4, 6, 3], n_classes)


def resnet50(n_classes=4):
    '''
    ResNet50.

    Args:
        n_classes: number of output parameters

    Returns:
        network ResNet50
    '''
    return ResNetDeep([3, 4, 6, 3], n_classes)


def resnet101(n_classes=4):
    '''
    ResNet101.

    Args:
        n_classes: number of output parameters

    Returns:
        network ResNet101
    '''
    return ResNetDeep([3, 4, 23, 3], n_classes)


def resnet152(n_classes=4):
    '''
    ResNet152.

    Args:
        n_classes: number of output parameters

    Returns:
        network ResNet152
    '''
    return ResNetDeep([3, 8, 36, 3], n_classes)
