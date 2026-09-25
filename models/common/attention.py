"""Common attention mechanisms."""

import torch
import torch.nn as nn


class SpatialAttentionModule(nn.Module):
    """Spatial attention module extracting max and avg pooled maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Keep the original HCFNet attribute name so published/source
        # checkpoints can be loaded without a key-rewrite shim.
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv2d(feat))
        return att * x
