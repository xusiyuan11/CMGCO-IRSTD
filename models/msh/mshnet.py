"""MSHNet: Multi-Scale Hierarchical Network for Infrared Small Target Detection.

CVPR 2024. Integrates CBAMResNet blocks with hierarchical feature fusion.
"""

import torch
import torch.nn as nn
from models.msh.cbam_resnet import CBAMResNetBlock


class MSHNet(nn.Module):
    """MSHNet architecture."""

    def __init__(self, in_features=3, out_features=1, block=CBAMResNetBlock, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.conv_init = nn.Conv2d(in_features, param_channels[0], 1, 1)

        # Encoders
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        # Decoders
        self.decoder_3 = self._make_layer(
            param_channels[3] + param_channels[4], param_channels[3], block, param_blocks[2]
        )
        self.decoder_2 = self._make_layer(
            param_channels[2] + param_channels[3], param_channels[2], block, param_blocks[1]
        )
        self.decoder_1 = self._make_layer(
            param_channels[1] + param_channels[2], param_channels[1], block, param_blocks[0]
        )
        self.decoder_0 = self._make_layer(
            param_channels[0] + param_channels[1], param_channels[0], block
        )

        # Multi-scale side outputs
        self.output_0 = nn.Conv2d(param_channels[0], out_features, 1)
        self.output_1 = nn.Conv2d(param_channels[1], out_features, 1)
        self.output_2 = nn.Conv2d(param_channels[2], out_features, 1)
        self.output_3 = nn.Conv2d(param_channels[3], out_features, 1)

        self.final = nn.Conv2d(4 * out_features, out_features, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layers = [block(in_channels, out_channels)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], dim=1))

        mask0 = self.output_0(x_d0)

        if self.deep_supervision:
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            final_out = self.final(
                torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1)
            )
            # return main final output and auxiliary outputs for deep supervision
            return final_out, mask0, mask1, mask2, mask3
        else:
            return mask0

