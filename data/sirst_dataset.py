"""SIRST Dataset loader supporting index splits and active-set curriculum."""

import os
import cv2
from torch.utils.data import Dataset
from data.transforms import get_train_transforms, get_test_transforms, get_eval_transforms


class SIRSTDataset(Dataset):
    """Dataset for Single-Frame Infrared Small Target Detection (SIRST)."""

    def __init__(
        self,
        data_root: str,
        idx_file: str = None,
        img_size: int = 512,
        mode: str = "train",
        in_channels: int = 3,
        preprocessing: str = "hcfnet",
    ):
        super().__init__()
        self.data_root = data_root
        self.img_size = int(img_size)
        self.mode = mode.lower()
        self.in_channels = int(in_channels)
        self.preprocessing = preprocessing.lower()

        self.image_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Load file list
        if idx_file:
            idx_path = os.path.join(data_root, idx_file) if not os.path.isabs(idx_file) else idx_file
            if os.path.isfile(idx_path):
                with open(idx_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                self.all_ids = []
                for line in lines:
                    base = os.path.splitext(line)[0]
                    self.all_ids.append(base)
            else:
                raise FileNotFoundError(f"Index file not found: {idx_path}")
        else:
            self.all_ids = self._scan_ids()

        self.active_ids = list(self.all_ids)

        if self.mode == "train":
            self.transform = get_train_transforms(self.img_size, self.preprocessing)
        else:
            self.transform = get_test_transforms(self.img_size, self.preprocessing)

        self.eval_transform = get_eval_transforms(self.img_size, self.preprocessing)

    def _scan_ids(self):
        if not os.path.isdir(self.image_dir):
            return []
        files = sorted(os.listdir(self.image_dir))
        ids = [os.path.splitext(f)[0] for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        return ids

    def set_active_ids(self, active_ids):
        """Update the active subset of images for curriculum learning."""
        active_set = set(active_ids)
        self.active_ids = [aid for aid in self.all_ids if aid in active_set]

    def reset_all_ids(self):
        """Reset active set to full dataset."""
        self.active_ids = list(self.all_ids)

    def __len__(self):
        return len(self.active_ids)

    def _find_file(self, folder, base_id):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG"):
            p = os.path.join(folder, base_id + ext)
            if os.path.isfile(p):
                return p
        return None

    def _load_raw_pair(self, base_id):
        img_path = self._find_file(self.image_dir, base_id)
        mask_path = self._find_file(self.mask_dir, base_id)

        if img_path is None or not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found for ID '{base_id}' in {self.image_dir}")

        image_flag = cv2.IMREAD_GRAYSCALE if self.in_channels == 1 else cv2.IMREAD_COLOR
        image = cv2.imread(img_path, image_flag)
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        if self.in_channels == 3:
            if self.preprocessing not in {"hcfnet", "samamba", "sam_mamba"}:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.in_channels != 1:
            raise ValueError(f"Unsupported input channel count: {self.in_channels}")

        if mask_path is None or not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask not found for ID '{base_id}' in {self.mask_dir}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask: {mask_path}")
        if self.preprocessing in {"msda", "msdanet"}:
            mask = (mask > 127).astype("uint8") * 255

        return image, mask

    def __getitem__(self, index):
        base_id = self.active_ids[index]
        image, mask = self._load_raw_pair(base_id)
        image_t, mask_t = self.transform(image, mask)
        return image_t, mask_t, base_id

    def get_eval_item(self, base_id):
        """Retrieve deterministic evaluation item (without random augmentations) for curriculum readiness."""
        image, mask = self._load_raw_pair(base_id)
        image_t, mask_t = self.eval_transform(image, mask)
        return image_t, mask_t, base_id

