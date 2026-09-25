"""Common convolutional building blocks."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard Convolution-Norm-Activation block."""

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        norm_type="bn",
        activation=True,
        use_bias=True,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=use_bias,
            groups=groups,
        )

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == "gn":
            num_groups = 32 if out_features >= 32 else out_features
            self.norm = nn.GroupNorm(num_groups, out_features)
        elif self.norm_type == "bn":
            self.norm = nn.BatchNorm2d(out_features)
        else:
            self.norm = None

        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x
