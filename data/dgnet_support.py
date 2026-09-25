"""DGNet dataset statistics and normalization from the local source release."""

from pathlib import Path

import numpy as np
from PIL import Image


DATASET_NORMALIZATION = {
    "NUAA-SIRST": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "SIRST": {"mean": 101.06385040283203, "std": 34.619606018066406},
    "NUDT-SIRST": {"mean": 107.80905151367188, "std": 33.02274703979492},
    "IRSTD-1K": {"mean": 87.4661865234375, "std": 39.71953201293945},
}


def normalize_image(image, config):
    return (image - config["mean"]) / config["std"]


def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name in DATASET_NORMALIZATION:
        return DATASET_NORMALIZATION[dataset_name].copy()
    dataset_dir = Path(dataset_dir)
    image_ids = []
    for split in ("train", "test"):
        path = dataset_dir / "img_idx" / f"{split}_{dataset_name}.txt"
        if path.is_file():
            image_ids.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not image_ids:
        raise FileNotFoundError(f"No train/test index files for {dataset_name} in {dataset_dir}")
    means, stds = [], []
    for image_id in image_ids:
        image_path = next((dataset_dir / "images" / f"{image_id}{ext}"
                           for ext in (".png", ".jpg", ".bmp")
                           if (dataset_dir / "images" / f"{image_id}{ext}").is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f"No image found for {image_id}")
        image = np.asarray(Image.open(image_path).convert("I"), dtype=np.float32)
        means.append(image.mean())
        stds.append(image.std())
    return {"mean": float(np.mean(means)), "std": float(np.mean(stds))}
