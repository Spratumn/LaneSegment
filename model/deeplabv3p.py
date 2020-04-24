import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F

from model.myconv import SeparableConv2d
from model.aspp import ASPPConv
from config import Config

cfg = Config()
group_num = cfg.NORMAL_GROUP_NUM


class XceptionBlock(nn.Module):
    """
    channel: no change
    size: 1/2
    """
    def __init__(self, in_channel, out_channels, normal='bn'):
        super(XceptionBlock, self).__init__()
        self.shortcut = nn.Conv2d(in_channel,
                                  out_channels[-1],
                                  kernel_size=1,
                                  stride=2,
                                  padding=0)
        layers = list()
        layers.append(nn.ReLU())  # this relu must be "inplace=False"
        layers.append(SeparableConv2d(in_channel,
                                      out_channels[0],
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False,
                                      normal=normal,
                                      act=0))
        for i in range(len(out_channels)-1):
            layers.append(SeparableConv2d(out_channels[i],
                                          out_channels[i+1],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=False,
                                          normal=normal,
                                          act=0))
        self.sep_conv1 = nn.Sequential(*layers)

        self.sep_conv2 = SeparableConv2d(out_channels[-2],
                                         out_channels[-1],
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         bias=False,
                                         normal=normal,
                                         act=0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        sep_conv1 = self.sep_conv1(x)
        sep_conv2 = self.sep_conv2(sep_conv1)
        return shortcut + sep_conv2, sep_conv1


class Decoder(nn.Module):
    def __init__(self, short_channel, normal='bn'):
        super(Decoder, self).__init__()
        self.shortcut_conv = nn.Conv2d(256,
                                       short_channel,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       bias=False)
        layers = list()
        layers.append(SeparableConv2d(256+short_channel,
                                      256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False,
                                      normal=normal,
                                      act=1))
        layers.append(SeparableConv2d(256, 256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False,
                                      normal=normal,
                                      act=1))
        self.decoder_conv = nn.Sequential(*layers)

    def forward(self, encoder_feature, shortcut):
        shortcut = self.shortcut_conv(shortcut)
        encoder_feature = F.interpolate(encoder_feature,
                                        size=(shortcut.shape[-2],
                                              shortcut.shape[-1]),
                                        mode='bilinear',
                                        align_corners=False)
        encoder_feature = torch.cat((encoder_feature, shortcut), 1)
        decoder_feature = self.decoder_conv(encoder_feature)
        return decoder_feature


class Deeplabv3plus(nn.Module):
    def __init__(self, in_channel=3, class_num=1000, normal='bn'):
        super(Deeplabv3plus, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.shortcut3 = XceptionBlock(64, [128, 128, 128], normal=normal)
        self.shortcut4 = XceptionBlock(128, [256, 256, 256], normal=normal)
        self.shortcut5 = XceptionBlock(256, [728, 728, 728], normal=normal)

        middle_layers = list()
        for i in range(16):
            middle_layers.append(SeparableConv2d(728, 728,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False,
                                                 normal=normal,
                                                 act=0))
        self.middle = nn.Sequential(*middle_layers)

        self.shortcut7 = XceptionBlock(728, [728, 1024, 1024], normal=normal)
        self.sep_conv8 = SeparableConv2d(1024, 1536,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=False,
                                         normal=normal,
                                         act=0)
        self.sep_conv9 = SeparableConv2d(1536, 1536,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         bias=False,
                                         normal=normal,
                                         act=0)
        self.sep_conv10 = SeparableConv2d(1536, 2048,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=False,
                                          normal=normal,
                                          act=0)
        
        self.encoder = ASPPConv(2048, 256, normal=normal)
        
        self.decoder = Decoder(short_channel=48, normal=normal)
        self.predictor = nn.Sequential(nn.Conv2d(256, class_num,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False))

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_shortcut3, _ = self.shortcut3(out_conv2)
        out_shortcut4, out_shortcut = self.shortcut4(out_shortcut3)
        out_shortcut5, _ = self.shortcut5(out_shortcut4)
        out_middle = self.middle(out_shortcut5)
        out_shortcut7, _ = self.shortcut7(out_middle)
        out_sep_conv8 = self.sep_conv8(out_shortcut7)
        out_sep_conv9 = self.sep_conv9(out_sep_conv8)
        out_sep_conv10 = self.sep_conv10(out_sep_conv9)

        encoder_feature = self.encoder(out_sep_conv10)
        decoder_feature = self.decoder(encoder_feature, out_shortcut)
        logit = self.predictor(decoder_feature)
        logit = F.interpolate(logit,
                              size=(x.shape[-2], x.shape[-1]),
                              mode='bilinear',
                              align_corners=False)
        return logit


if __name__ == '__main__':
    deeplabv3p = Deeplabv3plus(in_channel=3, class_num=8, normal='gn')
    summary(deeplabv3p, (3, 100, 100))



