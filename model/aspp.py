import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from model.myconv import AtrousConv

"""
ASPPConv(in_channel, out_channel, atrous_rates=[6,12,18])
"""


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channel, out_channel, normal='bn'):
        layers = [
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(in_channel, out_channel, 1, bias=False)
        ]
        if normal == 'bn':
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
        elif normal == 'gn':
            layers.append(nn.GroupNorm(2, out_channel))
            layers.append(nn.ReLU(inplace=True))
        super(ASPPPooling, self).__init__(*layers)

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPConv(nn.Module):
    def __init__(self, in_channel, out_channel, atrous_rates=(6, 12, 18), normal='bn'):
        super(ASPPConv, self).__init__()
        layers = list()
        layer_num = 2
        # 1x1 conv part
        conv_layer = list()
        conv_layer.append(nn.Conv2d(in_channel, out_channel, 1, bias=False))
        if normal == 'bn':
            conv_layer.append(nn.Sequential(nn.BatchNorm2d(out_channel)))
            conv_layer.append(nn.ReLU(inplace=True))
        elif normal == 'gn':
            conv_layer.append(nn.Sequential(nn.GroupNorm(2, out_channel)))
            conv_layer.append(nn.ReLU(inplace=True))
        layers.append(nn.Sequential(*conv_layer))
        # atrous conv part
        for atrous_rate in atrous_rates:
            layers.append(AtrousConv(in_channel, out_channel, atrous_rate, normal=normal))
            layer_num += 1
        # pooling part
        layers.append(ASPPPooling(in_channel, out_channel, normal))
        self.convs = nn.ModuleList(layers)

        project_layers = list()
        project_layers.append(nn.Conv2d(layer_num * out_channel, out_channel, 1, bias=False))
        if normal == 'bn':
            project_layers.append(nn.BatchNorm2d(out_channel))
            project_layers.append(nn.ReLU(inplace=True))
            project_layers.append(nn.Dropout(0.5))
        elif normal == 'gn':
            project_layers.append(nn.GroupNorm(2, out_channel))
            project_layers.append(nn.ReLU(inplace=True))
            project_layers.append(nn.Dropout(0.5))
        self.project = nn.Sequential(*project_layers)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


if __name__ == '__main__':
    asp = ASPPConv(2048, 256)
    summary(asp, (2048, 100, 100), device='cpu')
