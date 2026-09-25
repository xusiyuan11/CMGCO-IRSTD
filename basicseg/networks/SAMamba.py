"""SAMamba model adapter preserving the upstream network architecture."""

from pathlib import Path

import torch.nn as nn

from basicseg.main_blocks import CSI, DPCF
from basicseg.mona_with_select import MonaOp
from basicseg.networks.sam2.build_sam import build_sam2


class Adapter(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        self.monaOp = MonaOp(dim)

    def forward(self, x):
        return self.block(self.monaOp(x))


class SAMamba(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAMamba backbone checkpoint not found: {checkpoint_path}")

        model = build_sam2(
            "sam2_hiera_s.yaml",
            str(checkpoint_path),
            device="cpu",
        )
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.blocks = nn.Sequential(
            *(Adapter(block) for block in self.encoder.blocks)
        )

        self.mbhf1 = CSI(96, 128)
        self.mbhf2 = CSI(192, 128)
        self.mbhf3 = CSI(384, 128)
        self.mbhf4 = CSI(768, 128)

        self.up1 = DPCF(128, 128)
        self.up2 = DPCF(128, 128)
        self.up3 = DPCF(128, 128)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1 = self.mbhf1(x1)
        x2 = self.mbhf2(x2)
        x3 = self.mbhf3(x3)
        x4 = self.mbhf4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.head(self.deconv2(self.deconv1(x)))
