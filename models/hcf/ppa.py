"""Parallelized Patch-Aware Attention (PPA) module for HCFNet."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.conv import ConvBlock
from models.common.attention import SpatialAttentionModule


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) module."""

    def __init__(self, in_channel, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class LocalGlobalAttention(nn.Module):
    """Local-Global attention using patch unfold and prompt top-down transform."""

    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = nn.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = nn.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Patch unfolding
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # [B, H/P, W/P, P, P, C]
        local_patches = local_patches.reshape(B, -1, P * P, C)
        local_patches = local_patches.mean(dim=-1)  # [B, (H/P)*(W/P), P*P]

        local_patches = self.mlp1(local_patches)
        local_patches = self.norm(local_patches)
        local_patches = self.mlp2(local_patches)

        local_attention = F.softmax(local_patches, dim=-1)
        local_out = local_patches * local_attention

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode="bilinear", align_corners=False)
        output = self.conv(local_out)
        return output


class PPA(nn.Module):
    """Parallelized Patch-Aware Attention block."""

    def __init__(self, in_features, filters):
        super().__init__()
        self.skip = ConvBlock(
            in_features=in_features,
            out_features=filters,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type="bn",
            activation=False,
        )
        self.c1 = ConvBlock(
            in_features=in_features,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type="bn",
            activation=True,
        )
        self.c2 = ConvBlock(
            in_features=filters,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type="bn",
            activation=True,
        )
        self.c3 = ConvBlock(
            in_features=filters,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type="bn",
            activation=True,
        )
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
