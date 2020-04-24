import torch
import torch.nn as nn
from torchsummary import summary

from model.myconv import conv1x1, conv3x3, up_conv2x2, match_size, tensor_concat, SeparableConv2d
from config import Config

cfg = Config()


class UNet(nn.Module):
    """
    output size smaller than input size
    example: 572->388
    """
    def __init__(self, in_channel=3, class_num=2):
        super(UNet, self).__init__()
        self.conv_1 = nn.Sequential(conv3x3(in_channel, 64), conv3x3(64, 64))
        self.conv_2 = nn.Sequential(conv3x3(64, 128), conv3x3(128,  128))
        self.conv_3 = nn.Sequential(conv3x3(128, 256), conv3x3(256, 256))
        self.conv_4 = nn.Sequential(conv3x3(256, 512), conv3x3(512, 512))
        self.conv_5 = nn.Sequential(conv3x3(512, 1024), conv3x3(1024, 1024))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_conv_4 = nn.Sequential(up_conv2x2(1024, 512))
        self.conv_6 = nn.Sequential(conv3x3(1024, 512), conv3x3(512, 512))
        self.up_conv_3 = nn.Sequential(up_conv2x2(512, 256))
        self.conv_7 = nn.Sequential(conv3x3(512, 256), conv3x3(256, 256))
        self.up_conv_2 = nn.Sequential(up_conv2x2(256, 128))
        self.conv_8 = nn.Sequential(conv3x3(256, 128), conv3x3(128, 128))
        self.up_conv_1 = nn.Sequential(up_conv2x2(128, 64))
        self.conv_9 = nn.Sequential(conv3x3(128, 64), conv3x3(64, 64))
        self.result = conv1x1(64, class_num)
    
    def forward(self, x):
        stage_1 = self.conv_1(x)
        stage_2 = self.conv_2(self.maxpool(stage_1))
        stage_3 = self.conv_3(self.maxpool(stage_2))
        stage_4 = self.conv_4(self.maxpool(stage_3))

        up_in_4 = self.conv_5(self.maxpool(stage_4))
        up_stage_4 = self.up_conv_4(up_in_4)
        up_stage_4 = tensor_concat(up_stage_4, stage_4)

        up_in_3 = self.conv_6(up_stage_4)
        up_stage_3 = self.up_conv_3(up_in_3)
        up_stage_3 = tensor_concat(up_stage_3, stage_3)

        up_in_2 = self.conv_7(up_stage_3)
        up_stage_2 = self.up_conv_2(up_in_2)
        up_stage_2 = tensor_concat(up_stage_2, stage_2)

        up_in_1 = self.conv_8(up_stage_2)
        up_stage_1 = self.up_conv_1(up_in_1)
        up_stage_1 = tensor_concat(up_stage_1, stage_1)

        out = self.conv_9(up_stage_1)
        out = self.result(out)
        return out


class UNetv1(nn.Module):
    """
    output size same as input size
    example: 572->572
    """
    def __init__(self, in_channel=3, class_num=2, normal='bn'):
        super(UNetv1, self).__init__()

        # down sample stage
        self.conv_1 = nn.Sequential(conv3x3(in_channel, 64, padding=1, normal=normal),
                                    conv3x3(64, 64, padding=1, normal=normal))
        self.conv_2 = nn.Sequential(conv3x3(64, 128, padding=1, normal=normal),
                                    conv3x3(128, 128, padding=1, normal=normal))
        self.conv_3 = nn.Sequential(conv3x3(128, 256, padding=1, normal=normal),
                                    conv3x3(256, 256, padding=1, normal=normal))
        self.conv_4 = nn.Sequential(conv3x3(256, 512, padding=1, normal=normal),
                                    conv3x3(512, 512, padding=1, normal=normal))
        self.conv_5 = nn.Sequential(conv3x3(512, 1024, padding=1, normal=normal),
                                    conv3x3(1024, 1024, padding=1, normal=normal))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # upsample stage
        # up_conv_4 corresponds conv_4
        self.up_conv_4 = nn.Sequential(up_conv2x2(1024, 512))
        # conv the cat(stage_4,up_conv_4) from 1024 to 512
        self.conv_6 = nn.Sequential(conv3x3(1024, 512, padding=1, normal=normal),
                                    conv3x3(512, 512, padding=1, normal=normal))
        # up_conv_3 corresponds conv_3
        self.up_conv_3 = nn.Sequential(up_conv2x2(512, 256))
        # conv the cat(stage_3,up_conv_3) from 512 to 256
        self.conv_7 = nn.Sequential(conv3x3(512, 256, padding=1, normal=normal),
                                    conv3x3(256, 256, padding=1, normal=normal))
        # up_conv_2 corresponds conv_2
        self.up_conv_2 = nn.Sequential(up_conv2x2(256,128))
        # conv the cat(stage_2,up_conv_2) from 256 to 128
        self.conv_8 = nn.Sequential(conv3x3(256, 128, padding=1, normal=normal),
                                    conv3x3(128, 128, padding=1, normal=normal))
        # up_conv_1 corresponds conv_1
        self.up_conv_1 = nn.Sequential(up_conv2x2(128, 64))
        # conv the cat(stage_1,up_conv_1) from 128 to 64
        self.conv_9 = nn.Sequential(conv3x3(128, 64, padding=1, normal=normal),
                                    conv3x3(64, 64, padding=1, normal=normal))
        # output
        self.result = conv1x1(64, class_num, normal=normal)

    def forward(self, x):
        # get 4 stage conv output
        stage_1 = self.conv_1(x)
        size_1 = stage_1.shape[-2:]
        
        stage_2 = self.conv_2(self.maxpool(stage_1))
        size_2 = stage_2.shape[-2:]
        
        stage_3 = self.conv_3(self.maxpool(stage_2))
        size_3 = stage_3.shape[-2:]
        
        stage_4 = self.conv_4(self.maxpool(stage_3))
        size_4 = stage_4.shape[-2:]

        # get up_conv_4 and concat with stage_4
        up_in_4 = self.conv_5(self.maxpool(stage_4))
        up_stage_4 = self.up_conv_4(up_in_4)
        up_stage_4 = match_size(up_stage_4, size_4)
        up_stage_4 = torch.cat((stage_4, up_stage_4), 1)
        # get up_conv_3 and concat with stage_3
        up_in_3 = self.conv_6(up_stage_4)
        up_stage_3 = self.up_conv_3(up_in_3)
        up_stage_3 = match_size(up_stage_3, size_3)
        up_stage_3 = torch.cat((stage_3, up_stage_3), 1)
        # get up_conv_2 and concat with stage_2
        up_in_2 = self.conv_7(up_stage_3)
        up_stage_2 = self.up_conv_2(up_in_2)
        up_stage_2 = match_size(up_stage_2, size_2)
        up_stage_2 = torch.cat((stage_2, up_stage_2), 1)
        # get up_conv_1 and concat with stage_1
        up_in_1 = self.conv_8(up_stage_2)
        up_stage_1 = self.up_conv_1(up_in_1)
        up_stage_1 = match_size(up_stage_1, size_1)
        up_stage_1 = torch.cat((stage_1, up_stage_1), 1)

        # last conv to channel 2
        out = self.conv_9(up_stage_1)
        # result
        out = self.result(out)
        return out


class UNetv2(nn.Module):
    """
    use separable convolution
    output size same as input size
    example: 572->572
    """

    def __init__(self, in_channel=3, class_num=2, normal='bn'):
        super(UNetv2, self).__init__()

        # down sample stage
        self.conv_1 = nn.Sequential(SeparableConv2d(in_channel, 64, padding=1, normal='bn', act=1),
                                    SeparableConv2d(64, 64, padding=1, normal=normal, act=1))
        self.conv_2 = nn.Sequential(SeparableConv2d(64, 128, padding=1, normal=normal, act=1),
                                    SeparableConv2d(128, 128, padding=1, normal=normal, act=1))
        self.conv_3 = nn.Sequential(SeparableConv2d(128, 256, padding=1, normal=normal, act=1),
                                    SeparableConv2d(256, 256, padding=1, normal=normal, act=1))
        self.conv_4 = nn.Sequential(SeparableConv2d(256, 512, padding=1, normal=normal, act=1),
                                    SeparableConv2d(512, 512, padding=1, normal=normal, act=1))
        self.conv_5 = nn.Sequential(SeparableConv2d(512, 1024, padding=1, normal=normal, act=1),
                                    SeparableConv2d(1024, 1024, padding=1, normal=normal, act=1))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # upsample stage
        # up_conv_4 corresponds conv_4
        self.up_conv_4 = nn.Sequential(up_conv2x2(1024, 512))
        # conv the cat(stage_4,up_conv_4) from 1024 to 512
        self.conv_6 = nn.Sequential(SeparableConv2d(1024, 512, padding=1, normal=normal, act=1),
                                    SeparableConv2d(512, 512, padding=1, normal=normal, act=1))
        # up_conv_3 corresponds conv_3
        self.up_conv_3 = nn.Sequential(up_conv2x2(512, 256))
        # conv the cat(stage_3,up_conv_3) from 512 to 256
        self.conv_7 = nn.Sequential(SeparableConv2d(512, 256, padding=1, normal=normal, act=1),
                                    SeparableConv2d(256, 256, padding=1, normal=normal, act=1))
        # up_conv_2 corresponds conv_2
        self.up_conv_2 = nn.Sequential(up_conv2x2(256, 128))
        # conv the cat(stage_2,up_conv_2) from 256 to 128
        self.conv_8 = nn.Sequential(SeparableConv2d(256, 128, padding=1, normal=normal, act=1),
                                    SeparableConv2d(128, 128, padding=1, normal=normal, act=1))
        # up_conv_1 corresponds conv_1
        self.up_conv_1 = nn.Sequential(up_conv2x2(128, 64))
        # conv the cat(stage_1,up_conv_1) from 128 to 64
        self.conv_9 = nn.Sequential(SeparableConv2d(128, 64, padding=1, normal=normal, act=1),
                                    SeparableConv2d(64, 64, padding=1, normal=normal, act=1))
        # output
        self.result = conv1x1(64, class_num, normal=normal)

    def forward(self, x):
        # get 4 stage conv output
        stage_1 = self.conv_1(x)
        size_1 = stage_1.shape[-2:]

        stage_2 = self.conv_2(self.maxpool(stage_1))
        size_2 = stage_2.shape[-2:]

        stage_3 = self.conv_3(self.maxpool(stage_2))
        size_3 = stage_3.shape[-2:]

        stage_4 = self.conv_4(self.maxpool(stage_3))
        size_4 = stage_4.shape[-2:]

        # get up_conv_4 and concat with stage_4
        up_in_4 = self.conv_5(self.maxpool(stage_4))
        up_stage_4 = self.up_conv_4(up_in_4)
        up_stage_4 = match_size(up_stage_4, size_4)
        up_stage_4 = torch.cat((stage_4, up_stage_4), 1)
        # get up_conv_3 and concat with stage_3
        up_in_3 = self.conv_6(up_stage_4)
        up_stage_3 = self.up_conv_3(up_in_3)
        up_stage_3 = match_size(up_stage_3, size_3)
        up_stage_3 = torch.cat((stage_3, up_stage_3), 1)
        # get up_conv_2 and concat with stage_2
        up_in_2 = self.conv_7(up_stage_3)
        up_stage_2 = self.up_conv_2(up_in_2)
        up_stage_2 = match_size(up_stage_2, size_2)
        up_stage_2 = torch.cat((stage_2, up_stage_2), 1)
        # get up_conv_1 and concat with stage_1
        up_in_1 = self.conv_8(up_stage_2)
        up_stage_1 = self.up_conv_1(up_in_1)
        up_stage_1 = match_size(up_stage_1, size_1)
        up_stage_1 = torch.cat((stage_1, up_stage_1), 1)

        # last conv to channel 2
        out = self.conv_9(up_stage_1)
        # result
        out = self.result(out)
        return out


if __name__ == '__main__':
    # ut = UNetv1(in_channel=3, class_num=8, normal='bn')
    ut = UNetv2(in_channel=3, class_num=8, normal='gn')
    summary(ut, (3, 224, 224))
