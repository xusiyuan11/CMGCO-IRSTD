"""Multi-Dilated Channel Refiner (MDCR) module for HCFNet."""

import math
import torch
import torch.nn as nn
from models.common.conv import ConvBlock


class MDCR(nn.Module):
    """Multi-Dilated Channel Refiner with dilated group convolutions."""

    def __init__(
        self,
        in_features,
        out_features,
        norm_type="bn",
        activation=True,
        rate=(1, 6, 12, 18),
    ):
        super().__init__()
        quarter_in = in_features // 4
        quarter_out = out_features // 4
        groups = math.gcd(quarter_in, 128) if quarter_in >= 128 else quarter_in

        self.block1 = ConvBlock(
            in_features=quarter_in,
            out_features=quarter_out,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups=groups,
        )
        self.block2 = ConvBlock(
            in_features=quarter_in,
            out_features=quarter_out,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=groups,
        )
        self.block3 = ConvBlock(
            in_features=quarter_in,
            out_features=quarter_out,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=groups,
        )
        self.block4 = ConvBlock(
            in_features=quarter_in,
            out_features=quarter_out,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=groups,
        )
        self.out_s = ConvBlock(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = ConvBlock(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )

    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=1)
        x1 = self.block1(chunks[0])
        x2 = self.block2(chunks[1])
        x3 = self.block3(chunks[2])
        x4 = self.block4(chunks[3])

        # Mix channel-wise representations
        channel_count = x1.size(1)
        # Vectorized channel mixing to avoid slow python loops
        stacked = torch.stack([x1, x2, x3, x4], dim=2)  # [B, C, 4, H, W]
        B, C, G, H, W = stacked.shape
        stacked_reshaped = stacked.permute(0, 1, 2, 3, 4).reshape(B * C, 4, H, W)
        mixed = self.out_s(stacked_reshaped)  # [B*C, 4, H, W]
        mixed = mixed.reshape(B, C * 4, H, W)

        out = self.out(mixed)
        return out
