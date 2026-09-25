"""HCFNet: Hierarchical Context Fusion Network for Infrared Small Target Detection.

ICME 2024. Assembles PPA, DASI, and MDCR modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hcf.ppa import PPA
from models.hcf.dasi import DASI
from models.hcf.mdcr import MDCR


class HCFnet(nn.Module):
    """HCFNet architecture."""

    def __init__(
        self,
        in_features=3,
        out_features=1,
        base_channels=32,
        gt_ds=True,
    ):
        super().__init__()
        self.gt_ds = gt_ds
        c = base_channels

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Encoder stages
        self.p1 = PPA(in_features=in_features, filters=c)
        self.respath1 = DASI(in_features=c, out_features=c)

        self.p2 = PPA(in_features=c, filters=c * 2)
        self.respath2 = DASI(in_features=c * 2, out_features=c * 2)

        self.p3 = PPA(in_features=c * 2, filters=c * 4)
        self.respath3 = DASI(in_features=c * 4, out_features=c * 4)

        self.p4 = PPA(in_features=c * 4, filters=c * 8)
        self.respath4 = DASI(in_features=c * 8, out_features=c * 8)

        self.p5 = PPA(in_features=c * 8, filters=c * 16)
        self.mdcr = MDCR(in_features=c * 16, out_features=c * 16)

        # Decoder stages
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c * 16, c * 8, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )
        self.p6 = PPA(in_features=c * 8 * 2, filters=c * 8)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
        )
        self.p7 = PPA(in_features=c * 4 * 2, filters=c * 4)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
        )
        self.p8 = PPA(in_features=c * 2 * 2, filters=c * 2)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.p9 = PPA(in_features=c * 2, filters=c)

        # Output projection heads
        self.out = nn.Conv2d(c, out_features, kernel_size=1, padding=0)
        self.out1 = nn.Conv2d(c * 16, out_features, kernel_size=1, padding=0)
        self.out2 = nn.Conv2d(c * 8, out_features, kernel_size=1, padding=0)
        self.out3 = nn.Conv2d(c * 4, out_features, kernel_size=1, padding=0)
        self.out4 = nn.Conv2d(c * 2, out_features, kernel_size=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.p1(x)
        xp1 = self.maxpool(x1)

        x2 = self.p2(xp1)
        xp2 = self.maxpool(x2)

        x3 = self.p3(xp2)
        xp3 = self.maxpool(x3)

        x4 = self.p4(xp3)
        xp4 = self.maxpool(x4)

        x5 = self.p5(xp4)
        x_bottleneck = self.mdcr(x5)

        # Cross-scale feature paths via DASI
        x1_res = self.respath1(x1, x2, None)
        x2_res = self.respath2(x2, x3, x1)
        x3_res = self.respath3(x3, x4, x2)
        x4_res = self.respath4(x4, x_bottleneck, x3)

        # Decoder with deep supervision
        out4 = F.interpolate(self.out1(x_bottleneck), scale_factor=16, mode="bilinear", align_corners=True)

        d1 = self.up1(x_bottleneck)
        d1 = torch.cat([d1, x4_res], dim=1)
        d1 = self.p6(d1)
        out3 = F.interpolate(self.out2(d1), scale_factor=8, mode="bilinear", align_corners=True)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3_res], dim=1)
        d2 = self.p7(d2)
        out2 = F.interpolate(self.out3(d2), scale_factor=4, mode="bilinear", align_corners=True)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2_res], dim=1)
        d3 = self.p8(d3)
        out1 = F.interpolate(self.out4(d3), scale_factor=2, mode="bilinear", align_corners=True)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x1_res], dim=1)
        d4 = self.p9(d4)
        out = self.out(d4)

        if self.gt_ds:
            return out, out1, out2, out3, out4
        return out
