"""Evaluate a trained baseline or CMGCO checkpoint with the paired test protocol."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.builder import build_model
from train import DATASET_NAMES, PROJECT_ROOT, _build_dataset, _main_logits, _model_type, _resolve_dataset_config
from utils.logger import get_root_logger
from utils.metrics import PaperMetrics
from utils.misc import load_checkpoint
from utils.yaml_utils import parse_yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CMGCO detector")
    parser.add_argument("--opt", required=True, help="Training configuration YAML for this detector")
    parser.add_argument("--dataset", choices=sorted(DATASET_NAMES), help="Override dataset in YAML")
    parser.add_argument("--ckpt", required=True, help="Checkpoint from the corresponding paired run")
    parser.add_argument("--save-preds", help="Optional directory for thresholded masks")
    args = parser.parse_args()

    options = parse_yaml(args.opt)
    model_type = _model_type(options)
    dataset_name = _resolve_dataset_config(options, args.dataset)
    device = torch.device(
        f"cuda:{options.get('exp', {}).get('device', 0)}" if torch.cuda.is_available() else "cpu"
    )
    model = build_model(options["model"]).to(device)
    checkpoint = Path(args.ckpt)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    epoch, best_iou = load_checkpoint(model, str(checkpoint), device=device)
    test_dataset = _build_dataset(options["dataset"]["val"], model_type, "val")
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = None
    if args.save_preds:
        output_dir = Path(args.save_preds)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_root_logger()
    checkpoint_info = f" (epoch {epoch}, best validation IoU {best_iou:.2f}%)" if epoch is not None and best_iou is not None else " (source state dict; training metadata unavailable)"
    logger.info(f"Loaded {checkpoint}{checkpoint_info}")
    logger.info(f"Evaluating {model_type} on {dataset_name}: {len(test_dataset)} samples")
    model.eval()
    metrics = PaperMetrics(input_type="logits")
    with torch.no_grad():
        for images, masks, ids in tqdm(loader):
            predictions = model(images.to(device))
            logits = _main_logits(
                predictions, model_type, epoch,
                int(options.get("loss", {}).get("warm_epochs", 5)),
            )
            metrics.update(logits, masks.to(device))
            if output_dir is not None:
                binary = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(np.uint8) * 255
                for mask, sid in zip(binary, ids):
                    cv2.imwrite(str(output_dir / f"{sid}_pred.png"), mask[0])
    result = metrics.get()
    logger.info(
        f"IoU {result['IoU']:.2f}% | nIoU {result['nIoU']:.2f}% | "
        f"Pd {result['Pd']:.2f}% | Fa {result['Fa']:.2f} (10^-6)"
    )


if __name__ == "__main__":
    main()
