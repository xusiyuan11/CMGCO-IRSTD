"""Supervised loss functions for IRSTD models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    """Differentiable Soft IoU loss for binary mask prediction."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred: raw logits or probabilities
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        total = (pred + target).sum(dim=(1, 2, 3))
        union = total - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou.mean()


class SupervisedLoss(nn.Module):
    """Combines BCE loss and Soft IoU loss with deep supervision support."""

    def __init__(self, bce_weight=1.0, iou_weight=1.0, ds_weights=(1.0, 0.5, 0.25, 0.125, 0.0625)):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.iou_weight = float(iou_weight)
        self.ds_weights = ds_weights
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.iou_fn = SoftIoULoss()

    def _single_loss(self, pred, target):
        loss = 0.0
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce_fn(pred, target)
        if self.iou_weight > 0:
            loss = loss + self.iou_weight * self.iou_fn(pred, target)
        return loss

    def forward(self, preds, target):
        """Compute supervised loss.

        Args:
            preds: Output from network (Tensor [B, 1, H, W] or tuple of Tensors if deep supervision)
            target: Ground truth mask [B, 1, H, W]
        """
        if isinstance(preds, (list, tuple)):
            total_loss = 0.0
            for idx, pred in enumerate(preds):
                if pred.shape[-2:] != target.shape[-2:]:
                    pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
                w = self.ds_weights[idx] if idx < len(self.ds_weights) else 0.1
                total_loss = total_loss + w * self._single_loss(pred, target)
            return total_loss
        else:
            if preds.shape[-2:] != target.shape[-2:]:
                preds = F.interpolate(preds, size=target.shape[-2:], mode="bilinear", align_corners=False)
            return self._single_loss(preds, target)


class HCFNetSourceLoss(nn.Module):
    """Source BCE + IoU on the main output and four weighted side outputs."""

    def __init__(self, side_weights=(0.5, 0.25, 0.125, 0.0625)):
        super().__init__()
        self.side_weights = tuple(float(value) for value in side_weights)

    @staticmethod
    def _one(logits, target):
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        union = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou_loss = 1.0 - (intersection / (union + 1e-6)).mean()
        return F.binary_cross_entropy_with_logits(logits, target) + iou_loss

    def forward(self, preds, target, epoch=None):
        if not isinstance(preds, (tuple, list)) or len(preds) != 5:
            raise ValueError("HCFNet source supervision expects main plus four side outputs")
        return self._one(preds[0], target) + sum(
            weight * self._one(side, target)
            for weight, side in zip(self.side_weights, preds[1:])
        )


class MSHNetSourceLoss(nn.Module):
    """The local MSHNet SLSIoU objective and its five-epoch warm-up."""

    def __init__(self, warm_epochs=5):
        super().__init__()
        self.warm_epochs = int(warm_epochs)

    @staticmethod
    def _location_loss(probabilities, target):
        batch, _, height, width = probabilities.shape
        x_grid = torch.arange(width, device=probabilities.device, dtype=probabilities.dtype).view(1, 1, 1, width) / width
        y_grid = torch.arange(height, device=probabilities.device, dtype=probabilities.dtype).view(1, 1, height, 1) / height
        pred_x = (x_grid * probabilities).mean(dim=(1, 2, 3))
        pred_y = (y_grid * probabilities).mean(dim=(1, 2, 3))
        target_x = (x_grid * target).mean(dim=(1, 2, 3))
        target_y = (y_grid * target).mean(dim=(1, 2, 3))
        eps = 1e-8
        angle = (4.0 / torch.pi**2) * (
            torch.atan(pred_y / (pred_x + eps)) - torch.atan(target_y / (target_x + eps))
        ).square()
        pred_length = torch.sqrt(pred_x.square() + pred_y.square() + eps)
        target_length = torch.sqrt(target_x.square() + target_y.square() + eps)
        length_ratio = torch.minimum(pred_length, target_length) / (
            torch.maximum(pred_length, target_length) + eps
        )
        return (1.0 - length_ratio + angle).mean()

    def _one(self, logits, target, epoch):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        pred_sum = probabilities.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        eps = 1e-6
        overlap = (intersection + eps) / (pred_sum + target_sum - intersection + eps)
        if epoch <= self.warm_epochs:
            return 1.0 - overlap.mean()
        difference = ((pred_sum - target_sum) / 2.0).square()
        alpha = (torch.minimum(pred_sum, target_sum) + difference + eps) / (
            torch.maximum(pred_sum, target_sum) + difference + eps
        )
        return 1.0 - (alpha * overlap).mean() + self._location_loss(probabilities, target)

    def forward(self, preds, target, epoch):
        if not isinstance(preds, (tuple, list)) or len(preds) != 5:
            raise ValueError("MSHNet adapter expects main plus four side outputs")
        if epoch <= self.warm_epochs:
            return self._one(preds[1], target, epoch)
        result = self._one(preds[0], target, epoch)
        downsampled = target
        for index, side in enumerate(preds[1:]):
            if index > 0:
                downsampled = F.max_pool2d(downsampled, 2, 2)
            result = result + self._one(side, downsampled, epoch)
        return result / 5.0


class MSDANetSourceLoss(nn.Module):
    """Per-image SoftIoU from the local MSDA-Net training script."""

    def forward(self, logits, target, epoch=None):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        union = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        return 1.0 - ((intersection + 1.0) / (union + 1.0)).mean()


class DGNetSourceLoss(nn.Module):
    """Batch-global probability IoU from the local DGNet source."""

    def forward(self, logits, target, epoch=None):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum()
        union = probabilities.sum() + target.sum() - intersection
        return 1.0 - (intersection + 1.0) / (union + 1.0)


class SAMambaSourceLoss(nn.Module):
    """Unweighted SoftIoU + WeightedDice + Focal objectives from SAMamba."""

    def forward(self, logits, target, epoch=None):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum(dim=(1, 2, 3))
        union = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        soft_iou = 1.0 - ((intersection + 1e-6) / (union + 1e-6)).mean()

        # The source WeightedDiceLoss uses a unit weight map by default.
        dice_numerator = 2.0 * intersection
        dice_denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        weighted_dice = 1.0 - ((dice_numerator + 1e-6) / (dice_denominator + 1e-6)).mean()

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        focal = (0.25 * (1.0 - torch.exp(-bce)).square() * bce).mean()
        return soft_iou + weighted_dice + focal


def build_source_supervised_loss(model_type, config=None):
    config = config or {}
    model_type = model_type.lower()
    if model_type in {"hcfnet", "hcf"}:
        return HCFNetSourceLoss(side_weights=config.get("side_weights", (0.5, 0.25, 0.125, 0.0625)))
    if model_type in {"mshnet", "msh"}:
        return MSHNetSourceLoss(warm_epochs=config.get("warm_epochs", 5))
    if model_type in {"msdanet", "msda", "msda_net"}:
        return MSDANetSourceLoss()
    if model_type in {"dgnet", "dg"}:
        return DGNetSourceLoss()
    if model_type in {"samamba", "sam_mamba"}:
        return SAMambaSourceLoss()
    raise ValueError(f"No source supervised objective registered for {model_type}")

