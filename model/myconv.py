import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Config

cfg = Config()

group_num = cfg.NORMAL_GROUP_NUM
"""
SeparableConv2d(in_channel, out_channel)
AtrousConv(in_channel, out_channel, dilation)
"""


def conv3x3(in_planes, out_planes, padding=0, stride=1, normal='bn', rl=True):
    """3x3 convolution with padding"""
    layers = list()
    layers.append(nn.Conv2d(in_planes,
                            out_planes,
                            kernel_size=3, 
                            padding=padding, 
                            stride=stride))
    if normal == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
    elif normal == 'gn':
        layers.append(nn.GroupNorm(group_num, out_planes))
    if rl:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv1x1(in_planes, out_planes, stride=1, normal='bn', rl=True):
    """1x1 convolution without padding"""
    layers = list()
    layers.append(nn.Conv2d(in_planes, 
                            out_planes,
                            kernel_size=1,
                            padding=0,
                            stride=stride))
    if normal == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
    elif normal == 'gn':
        layers.append(nn.GroupNorm(group_num, out_planes))
    if rl:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def up_conv2x2(in_planes, out_planes):
    return nn.ConvTranspose2d(in_planes,
                              out_planes,
                              kernel_size=2,
                              stride=2)


def match_size(src_tensor, dst_size):
    """
    make the src_tensor shape of dst_size
    """
    return F.interpolate(src_tensor, 
                         size=dst_size,
                         mode='bilinear',
                         align_corners=False)


def tensor_concat(small_tensor, big_tensor):
    """
    concat two tensor by the channel axes
    tensor shape of (n,c,h,w)
    """
    h_crop = big_tensor.size()[2]-small_tensor.size()[2]
    w_crop = big_tensor.size()[3]-small_tensor.size()[3]
    if not h_crop % 2 == 0:
        h_crop -= 1
        h_crop /= 2
        h_crop = int(h_crop)
    if not w_crop % 2 == 0:
        w_crop /= 2
        w_crop = int(w_crop)

    big_tensor = big_tensor[:, :, h_crop:small_tensor.size()[2]+h_crop,
                            w_crop:small_tensor.size()[3]+w_crop]
    return torch.cat((big_tensor, small_tensor), 1)


class SeparableConv2d(nn.Sequential):
    """
    define separable Convolution Sequential
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 normal='bn',
                 act=0):
        layers = list()
        layers.append(nn.Conv2d(in_channel,
                                in_channel,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=in_channel,
                                bias=bias))
        if normal == 'bn':
            layers.append(nn.BatchNorm2d(in_channel))
        elif normal == 'gn':
            layers.append(nn.GroupNorm(group_num, in_channel))
        if act == 1:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channel,
                                out_channel,
                                kernel_size=1,
                                bias=False))
        if act == 1:
            layers.append(nn.ReLU(inplace=True))
        super(SeparableConv2d, self).__init__(*layers)


class AtrousConv(nn.Sequential):
    """
    define atrous Convolution Sequential with specific atrous_rate
    kernel_size = 3  -->  padding = dilation
    stride = 1
    channels no change: out_channel = in_channel
    out_size no change
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 dilation,
                 normal='bn'):
        layers = [
            SeparableConv2d(in_channel,
                            out_channel,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False,
                            normal=normal,
                            act=0)
                ]
        if normal == 'bn':
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
        elif normal == 'gn':
            layers.append(nn.GroupNorm(group_num, out_channel))
            layers.append(nn.ReLU(inplace=True))

        super(AtrousConv, self).__init__(*layers)
