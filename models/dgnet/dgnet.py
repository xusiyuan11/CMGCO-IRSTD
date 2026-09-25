"""DGNet: Dual-knowledge Guided Network for Infrared Small Target Detection.

ACM MM 2026.
"""

from pathlib import Path

import torch
import torch.nn as nn
from models.dgnet.attention import ResidualAttentionBlock
from models.dgnet.pwm import PriorKnowledgeWaveletModulation


class DGNet(nn.Module):
    """Dual-knowledge Guided Network."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        block=ResidualAttentionBlock,
        num_blocks=(2, 2, 2, 2),
        nb_filter=(16, 32, 64, 128, 256),
        text_features_path=None,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up8 = nn.Upsample(scale_factor=8)
        self.up16 = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, in_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.dwt0 = PriorKnowledgeWaveletModulation(nb_filter[0], nb_filter[0], wt_type="haar")
        self.dwt1 = PriorKnowledgeWaveletModulation(nb_filter[1], nb_filter[1], wt_type="haar")
        self.dwt2 = PriorKnowledgeWaveletModulation(nb_filter[2], nb_filter[2], wt_type="haar")
        self.dwt3 = PriorKnowledgeWaveletModulation(nb_filter[3], nb_filter[3], wt_type="haar")

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

        if text_features_path is None:
            raise ValueError("DGNet requires the fixed text_features_path asset")
        feature_path = Path(text_features_path)
        if not feature_path.is_file():
            raise FileNotFoundError(f"DGNet text features not found: {feature_path}")
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
        target_features = features["target"].float().reshape(1, -1)
        background_features = features["background"].float().reshape(1, -1)
        if target_features.shape[1] != 512 or background_features.shape[1] != 512:
            raise ValueError(
                f"DGNet expects two 512-dimensional text features, got "
                f"{tuple(target_features.shape)} and {tuple(background_features.shape)}"
            )
        self.register_buffer("text_eot_targets", target_features, persistent=False)
        self.register_buffer("text_eot_bg", background_features, persistent=False)

    def _make_layer(self, block, in_channels, out_channels, num_blocks=1):
        layers = [block(in_channels, out_channels)]
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input_tensor, text_eot_targets=None, text_eot_bg=None):
        B = input_tensor.shape[0]
        if text_eot_targets is None:
            text_eot_targets = self.text_eot_targets.expand(B, -1)
        if text_eot_bg is None:
            text_eot_bg = self.text_eot_bg.expand(B, -1)

        x0_0 = self.conv0_0(input_tensor)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0d_0 = self.dwt0(x0_0, text_eot_targets, text_eot_bg)
        x1d_0 = self.dwt1(x1_0, text_eot_targets, text_eot_bg)
        x2d_0 = self.dwt2(x2_0, text_eot_targets, text_eot_bg)
        x3d_0 = self.dwt3(x3_0, text_eot_targets, text_eot_bg)

        xu3 = self.conv0_4_1x1(self.up16(x4_0))
        x3_1 = self.conv4_1(torch.cat([x3d_0, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))

        x2_1 = self.conv3_1(torch.cat([x2d_0, self.up(x3_1)], 1))
        xu1 = self.conv0_2_1x1(self.up4(x2_1))

        x1_1 = self.conv2_1(torch.cat([x1d_0, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0d_0, self.up(x1_1)], 1))

        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)
        output = self.final(xf)
        return output

