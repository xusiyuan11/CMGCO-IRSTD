"""Receptive Field Block (RFB) and attention blocks for MSDANet."""

import torch
import torch.nn as nn
from models.msdanet.wavelet_attn import (
    LH_DWT_2D_attation,
    HL_DWT_2D_attation,
    HH_DWT_2D_attation,
    LL_DWT_2D_attation,
)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 4), channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        return self.sigmoid(self.conv(result))


class FrequencyAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.HH_reduce = SpatialAttention()
        self.LH_reduce = SpatialAttention()
        self.HL_reduce = SpatialAttention()
        self.LL_reduce = SpatialAttention()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channel),
        )
        self.relu = nn.ReLU(inplace=False)
        # Preserve the source module names for direct checkpoint compatibility.
        self.LH_DWT_2D_attation = LH_DWT_2D_attation('haar')
        self.HL_DWT_2D_attation = HL_DWT_2D_attation('haar')
        self.HH_DWT_2D_attation = HH_DWT_2D_attation('haar')
        self.LL_DWT_2D_attation = LL_DWT_2D_attation('haar')

    def forward(self, x):
        out_LH = x * self.LH_reduce(self.LH_DWT_2D_attation(x))
        out_HL = x * self.HL_reduce(self.HL_DWT_2D_attation(x))
        out_HH = x * self.HH_reduce(self.HH_DWT_2D_attation(x))
        out_LL = x * self.LL_reduce(self.LL_DWT_2D_attation(x))
        x_out = out_LH + out_HL + out_HH + out_LL
        return self.relu(x + self.conv_res(x_out))


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=5, padding=2),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.FrequencyAttention = FrequencyAttention(in_channel=out_channel)
        self.SEAttention = SEAttention(channel=out_channel, reduction=4)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x_cat = self.FrequencyAttention(x_cat)
        x_cat = self.SEAttention(x_cat)
        return self.relu(x + self.conv_res(x_cat))


class RFB_modified_LCL(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=5, padding=2),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        return self.relu(x_cat)

