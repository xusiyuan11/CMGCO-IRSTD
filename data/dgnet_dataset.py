"""Curriculum-aware adapter around the local DGNet source data pipeline."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.dgnet_source import InfraredSmallTargetDataset
from data.dgnet_support import normalize_image


class DGNetDataset(Dataset):
    def __init__(self, data_root, img_size=256, mode="train"):
        root = Path(data_root)
        if int(img_size) != 256:
            raise ValueError("The local DGNet protocol uses 256x256 inputs")
        self.mode = mode.lower()
        self.source = InfraredSmallTargetDataset(
            dataset_dir=root.parent,
            dataset_name=root.name,
            patch_size=256,
            mode="train" if self.mode == "train" else "val",
            use_aug=self.mode == "train",
        )
        self.all_ids = list(self.source.files)
        self.active_ids = list(self.all_ids)
        self._positions = {sid: index for index, sid in enumerate(self.all_ids)}
        if len(self._positions) != len(self.all_ids):
            raise ValueError("DGNet split contains duplicate image IDs")

    def set_active_ids(self, active_ids):
        active = set(active_ids)
        self.active_ids = [sid for sid in self.all_ids if sid in active]

    def __len__(self):
        return len(self.active_ids)

    def __getitem__(self, index):
        sid = self.active_ids[index]
        result = self.source[self._positions[sid]]
        if self.mode == "train":
            original, augmented, mask = result
            return original, augmented, mask, sid
        image, mask, _, _ = result
        return image, mask, sid

    def get_eval_item(self, sid):
        if sid not in self._positions:
            raise KeyError(sid)
        image, mask = self.source._load_pair(sid)
        image = image.resize((256, 256), Image.Resampling.NEAREST)
        mask = mask.resize((256, 256), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32)
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        if mask_array.ndim > 2:
            mask_array = mask_array[:, :, 0]
        image_t = torch.from_numpy(np.ascontiguousarray(
            normalize_image(image_array, self.source.img_norm_cfg)[None, :, :]
        ))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_array[None, :, :]))
        return image_t, mask_t, sid
