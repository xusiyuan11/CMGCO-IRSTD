"""MSDANet: Multi-Scale Direction-Aware Network for Infrared Small Target Detection.

TGRS 2025.
"""

import torch
import torch.nn as nn
from models.msdanet.wavelet_attn import Hfrequency
from models.msdanet.rfb import RFB_modified, RFB_modified_LCL
from models.msdanet.sbam import Resnet1, Resnet2, Sbam


class Stage(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )
        self.resnet1_1 = Resnet1(in_channel=16, out_channel=16)
        self.resnet1_2 = RFB_modified(in_channel=16, out_channel=16)
        self.resnet1_3 = RFB_modified(in_channel=16, out_channel=16)
        self.layer1_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )

        self.resnet2_1 = Resnet2(in_channel=16, out_channel=23)
        self.hfrequency = Hfrequency()
        self.resnet2_2 = RFB_modified(in_channel=32, out_channel=32)
        self.resnet2_3 = RFB_modified(in_channel=32, out_channel=32)
        self.layer2_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )

        self.resnet3_1 = Resnet2(in_channel=32, out_channel=64)
        self.resnet3_2 = RFB_modified(in_channel=64, out_channel=64)
        self.resnet3_3 = RFB_modified(in_channel=64, out_channel=64)
        self.layer3_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )

        self.resnet4_1 = Resnet2(in_channel=64, out_channel=64)
        self.resnet4_2 = RFB_modified(in_channel=64, out_channel=64)
        self.resnet4_3 = RFB_modified(in_channel=64, out_channel=64)
        self.layer4_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )

        self.resnet5_1 = Resnet2(in_channel=64, out_channel=64)
        self.resnet5_2 = RFB_modified(in_channel=64, out_channel=64)
        self.resnet5_3 = RFB_modified(in_channel=64, out_channel=64)
        self.layer5_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        outs = []
        out = self.layer1(x)

        out = self.resnet1_1(out)
        out = self.resnet1_2(out)
        out = self.resnet1_3(out)
        out_1 = self.layer1_4(out)
        outs.append(out)

        out = self.resnet2_1(out)
        out = self.hfrequency(x, out)
        out = self.resnet2_2(out)
        out = self.resnet2_3(out)
        out_2 = self.layer2_4(out)
        outs.append(out)

        out = self.resnet3_1(out)
        out = self.resnet3_2(out)
        out = self.resnet3_3(out)
        out_3 = self.layer3_4(out)
        outs.append(out)

        out = self.resnet4_1(out)
        out = self.resnet4_2(out)
        out = self.resnet4_3(out)
        out_4 = self.layer4_4(out)
        outs.append(out)

        out = self.resnet5_1(out)
        out = self.resnet5_2(out)
        out = self.resnet5_3(out)
        out = torch.cat([out, out_4, out_3, out_2, out_1], dim=1)
        out = self.layer5_4(out)
        outs.append(out)

        return outs


class MSDANet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.stage = Stage(in_channels=in_channels)
        self.mlcl5 = RFB_modified_LCL(64, 64)
        self.mlcl4 = RFB_modified_LCL(64, 64)
        self.mlcl3 = RFB_modified_LCL(64, 64)
        self.mlcl2 = RFB_modified_LCL(32, 32)
        self.mlcl1 = RFB_modified_LCL(16, 16)

        self.sbam4 = Sbam(64, 64)
        self.sbam3 = Sbam(64, 64)
        self.sbam2 = Sbam(64, 32)
        self.sbam1 = Sbam(32, 16)

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        outs = self.stage(x)

        out5 = self.mlcl5(outs[4])
        out4 = self.mlcl4(outs[3])
        out3 = self.mlcl3(outs[2])
        out2 = self.mlcl2(outs[1])
        out1 = self.mlcl1(outs[0])

        out4_2 = self.sbam4(out5, out4)
        out3_2 = self.sbam3(out4_2, out3)
        out2_2 = self.sbam2(out3_2, out2)
        out1_2 = self.sbam1(out2_2, out1)

        out = self.layer(out1_2)
        return out

