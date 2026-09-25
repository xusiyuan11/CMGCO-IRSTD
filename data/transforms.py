"""Detector-specific preprocessing used by the paired CMGCO experiments.

The unified trainer keeps one dataset interface, but each detector retains the
preprocessing used by its local source implementation. Readiness evaluation
always uses the matching deterministic test transform.
"""

import inspect
import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter, ImageOps


class AlbumentationsPairTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, mask=None):
        result = self.transform(image=image, mask=mask)
        image_t = result["image"].float()
        mask_t = result["mask"].float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        if mask_t.max() > 1.0:
            mask_t = mask_t / 255.0
        return image_t, mask_t


def _compose_pair(transforms):
    """Keep the source transform list compatible with older Albumentations."""
    options = {}
    if "is_check_shapes" in inspect.signature(A.Compose).parameters:
        options["is_check_shapes"] = False
    return AlbumentationsPairTransform(A.Compose(transforms, **options))


class MSHNetTrainTransform:
    """MSHNet random scale/crop/blur and ImageNet normalization."""

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, base_size=512, crop_size=512):
        self.base_size = int(base_size)
        self.crop_size = int(crop_size)

    @staticmethod
    def _to_tensor(image, mask):
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        image_t = (image_t - MSHNetTrainTransform.mean) / MSHNetTrainTransform.std
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        return image_t, mask_t

    def __call__(self, image, mask=None):
        image = Image.fromarray(image).convert("RGB")
        mask = Image.fromarray(mask).convert("L")
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = image.size
        if height > width:
            out_h = long_size
            out_w = int(width * long_size / height + 0.5)
            short_size = out_w
        else:
            out_w = long_size
            out_h = int(height * long_size / width + 0.5)
            short_size = out_h
        image = image.resize((out_w, out_h), Image.Resampling.BILINEAR)
        mask = mask.resize((out_w, out_h), Image.Resampling.NEAREST)

        if short_size < self.crop_size:
            pad_h = max(0, self.crop_size - out_h)
            pad_w = max(0, self.crop_size - out_w)
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

        width, height = image.size
        x0 = random.randint(0, width - self.crop_size)
        y0 = random.randint(0, height - self.crop_size)
        box = (x0, y0, x0 + self.crop_size, y0 + self.crop_size)
        image = image.crop(box)
        mask = mask.crop(box)
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return self._to_tensor(image, mask)


class MSHNetTestTransform(MSHNetTrainTransform):
    def __call__(self, image, mask=None):
        image = Image.fromarray(image).convert("RGB")
        mask = Image.fromarray(mask).convert("L")
        image = image.resize((self.base_size, self.base_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.Resampling.NEAREST)
        return self._to_tensor(image, mask)


def _hcf_train(img_size):
    return _compose_pair([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.Resize(height=img_size, width=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def _hcf_test(img_size):
    return _compose_pair([
        A.Resize(height=img_size, width=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def _msdanet_train(img_size):
    return _compose_pair([
        A.Resize(height=img_size, width=img_size),
        A.SomeOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.2),
            A.RandomContrast(limit=0.2, p=0.2),
            A.Rotate(limit=45, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.2),
            A.NoOp(),
            A.NoOp(),
        ], n=3, p=0.5),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def _msdanet_test(img_size):
    return _compose_pair([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_transforms(profile="hcfnet", img_size=512, mode="train"):
    profile = profile.lower()
    is_train = mode.lower() == "train"
    if profile in {"hcf", "hcfnet"}:
        return _hcf_train(img_size) if is_train else _hcf_test(img_size)
    if profile in {"samamba", "sam_mamba"}:
        # The local SAMamba Dataset_aug_bac delegates to aug_transform_train,
        # which uses these same augmentation and test-transform lists.
        return _hcf_train(img_size) if is_train else _hcf_test(img_size)
    if profile in {"msh", "mshnet"}:
        cls = MSHNetTrainTransform if is_train else MSHNetTestTransform
        return cls(base_size=img_size, crop_size=img_size)
    if profile in {"msda", "msdanet"}:
        return _msdanet_train(img_size) if is_train else _msdanet_test(img_size)
    raise ValueError(f"Unknown preprocessing profile: {profile}")


def get_train_transforms(img_size=512, profile="hcfnet"):
    return build_transforms(profile, img_size, "train")


def get_test_transforms(img_size=512, profile="hcfnet"):
    return build_transforms(profile, img_size, "test")


def get_eval_transforms(img_size=512, profile="hcfnet"):
    return build_transforms(profile, img_size, "eval")
