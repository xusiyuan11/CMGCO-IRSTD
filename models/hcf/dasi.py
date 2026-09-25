"""Dimension-Aware Selective Integration (DASI) and Bag modules for HCFNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.conv import ConvBlock


class Bag(nn.Module):
    """Boundary-aware gating fusion."""

    def __init__(self):
        super().__init__()

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1.0 - edge_att) * i


class DASI(nn.Module):
    """Dimension-Aware Selective Integration module."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.bag = Bag()
        self.tail_conv = nn.Sequential(
            ConvBlock(
                in_features=out_features,
                out_features=out_features,
                kernel_size=(1, 1),
                padding=(0, 0),
                norm_type=None,
                activation=False,
            )
        )
        self.conv = nn.Sequential(
            ConvBlock(
                in_features=out_features // 2,
                out_features=out_features // 4,
                kernel_size=(1, 1),
                padding=(0, 0),
                norm_type=None,
                activation=False,
            )
        )
        self.bns = nn.BatchNorm2d(out_features)

        self.skips = ConvBlock(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=None,
            activation=False,
        )
        self.skips_2 = ConvBlock(
            in_features=in_features * 2,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=None,
            activation=False,
        )
        self.skips_3 = nn.Conv2d(
            in_features // 2, out_features, kernel_size=3, stride=2, dilation=2, padding=2
        )
        self.relu = nn.ReLU()

    def forward(self, x, x_low, x_high):
        if x_high is not None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low is not None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(
                x_low, size=[x.size(2), x.size(3)], mode="bilinear", align_corners=True
            )
            x_low = torch.chunk(x_low, 4, dim=1)

        x_skip = self.skips(x)
        x = self.skips(x)
        x = torch.chunk(x, 4, dim=1)

        if x_high is None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low is None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.tail_conv(out)
        out += x_skip
        out = self.bns(out)
        out = self.relu(out)
        return out
