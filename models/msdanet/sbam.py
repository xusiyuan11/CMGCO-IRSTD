"""Residual blocks and Sbam integration module for MSDANet."""

import torch
import torch.nn as nn


class Resnet1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out = out + identity
        return self.relu(out)


class Resnet2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        identity = self.layer2(identity)
        out = out + identity
        return self.relu(out)


class Sbam(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.hl_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.hl_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
        )
        self.concat_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel + out_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=False),
        )

    def forward(self, hl, ll):
        hl = self.hl_up(hl)
        concat = torch.cat((hl, ll), 1)
        k = self.concat_layer(concat)
        hl = hl + k
        hl = self.hl_layer(hl)
        return ll + hl

