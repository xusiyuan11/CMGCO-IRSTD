"""Miscellaneous helper functions (seed, checkpointing, device handling)."""

import os
import random
import numpy as np
import torch


def set_seed(seed=42, cuda_deterministic=False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, save_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    state = {
        "epoch": epoch,
        "best_iou": best_iou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, save_path)


def load_checkpoint(model, load_path, optimizer=None, scheduler=None, device="cpu"):
    """Load model checkpoint."""
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"Checkpoint not found at: {load_path}")

    checkpoint = torch.load(load_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Clean potential DDP 'module.' prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    model.load_state_dict(cleaned_state_dict, strict=True)

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch")
    best_iou = checkpoint.get("best_iou")
    return epoch, best_iou
