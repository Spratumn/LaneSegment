import torch
import torch.nn as nn
from torchsummary import summary

from model.myconv import conv3x3, conv1x1
from config import Config

cfg = Config()

group_num = cfg.NORMAL_GROUP_NUM


class BasicBlock(nn.Module):
    """
    apply in shallow net: resnet18,resnet34
    
    conv3x3 (bn+rl)
    conv3x3 (bn)
    relu(bn(x+shortcut))
    """
    expansion = 1

    def __init__(self, inchannel, channel, stride=1, downsample=None, normal='bn'):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = conv3x3(inchannel, channel, padding=1, stride=stride, normal=normal)
        self.conv2 = conv3x3(channel, channel * self.expansion, 
                             padding=1, normal=None, rl=False)
        if normal == 'bn':
            self.normal = nn.BatchNorm2d(channel*self.expansion)
        elif normal == 'gn':
            self.normal = nn.GroupNorm(group_num, channel * self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.relu(self.normal(x))
        return x


class Bottleneck(nn.Module):
    """
    apply in deep net: : resnet50,resnet101,resnet152
    conv1x1 (bn+rl)
    conv3x3 (bn+rl)
    conv1x1 (bn)
    relu(bn(x+shortcut))
    """
    expansion = 4

    def __init__(self, inchannel, channel, stride=1, downsample=None, normal='bn'):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = conv1x1(inchannel, channel, stride=stride, normal=normal)
        self.conv2 = conv3x3(channel, channel, padding=1, normal=normal)
        self.conv3 = conv1x1(channel, channel * self.expansion, 
                             normal=None, rl=False)
        if normal == 'bn':
            self.normal = nn.BatchNorm2d(channel*self.expansion)
        elif normal == 'gn':
            self.normal = nn.GroupNorm(group_num, channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.relu(self.normal(x))
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, class_num=1000, normal='bn'):
        super(ResNet, self).__init__()
        self.inchannel_num = 64
        self.conv = nn.Conv2d(3, self.inchannel_num, kernel_size=7, stride=2, padding=3, bias=False)
        if normal == 'bn':
            self.normal = nn.BatchNorm2d(self.inchannel_num)
        elif normal == 'gn':
            self.normal = nn.GroupNorm(group_num, self.inchannel_num)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], change_size=False, normal=normal)
        self.layer2 = self._make_layer(block, 128, layers[1], change_size=True, normal=normal)
        self.layer3 = self._make_layer(block, 256, layers[2], change_size=True, normal=normal)
        self.layer4 = self._make_layer(block, 512, layers[3], change_size=True, normal=normal)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, class_num)
    
    def _make_layer(self, block, channel_num, block_num, change_size=False, normal='bn'):
        # change_size=True means the first block need change input size
        if change_size:
            stride = 2
        else:
            stride = 1
        if block is BasicBlock:
            if change_size:
                downsample = conv3x3(self.inchannel_num,
                                     channel_num*block.expansion,
                                     padding=1,
                                     stride=stride,
                                     normal=normal)
            else:
                downsample = None
        elif block is Bottleneck:
            downsample = conv1x1(self.inchannel_num,
                                 channel_num*block.expansion,
                                 stride=stride,
                                 normal=normal)
        else:
            raise ValueError('"block" should be "BasicBlock" or "Bottleneck"')
        layers = list()
        layers.append(block(self.inchannel_num,
                            channel_num,
                            stride=stride,
                            downsample=downsample,
                            normal=normal))
        for _ in range(1, block_num):
            layers.append(block(channel_num*block.expansion,
                                channel_num,
                                normal=normal))
        self.inchannel_num = channel_num * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.normal(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# define net create function
def resnet18(class_num=1000, normal='bn'):
    return ResNet(BasicBlock, [2, 2, 2, 2], class_num, normal=normal)


def resnet34(class_num=1000, normal='bn'):
    return ResNet(BasicBlock, [3, 4, 6, 3], class_num, normal=normal)


def resnet50(class_num=1000, normal='bn'):
    return ResNet(Bottleneck, [3, 4, 6, 3], class_num, normal=normal)


def resnet101(class_num=1000, normal='bn'):
    return ResNet(Bottleneck, [3, 4, 23, 3], class_num, normal=normal)


def resnet152(class_num=1000, normal='bn'):
    return ResNet(Bottleneck, [3, 8, 36, 3], class_num, normal=normal)


if __name__ == '__main__':
    rs = resnet50(normal='gn')
    summary(rs, (3, 224, 224))

