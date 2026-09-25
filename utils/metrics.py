"""Paper evaluation metrics with source-majority component matching for F_a.

Strictly follows Section IV-B (Eq. 20-22) of the CMGCO manuscript:
- IoU: Aggregates foreground intersections and unions over the complete test set.
- nIoU: Normalized IoU averaging image-wise overlap.
- P_d: Target-level detection probability using centroid matching (distance < 3 pixels).
- F_a: Unmatched predicted-component pixel area / N_pix, in units of 10^-6.
"""

import numpy as np
import torch
from skimage import measure


class PaperMetrics:
    """Evaluator using the manuscript formulas and local source matching convention."""

    def __init__(self, threshold=0.5, match_distance=3.0, eps=1e-12, input_type="logits"):
        self.threshold = float(threshold)
        self.match_distance = float(match_distance)
        self.eps = float(eps)
        if input_type not in {"logits", "probabilities"}:
            raise ValueError(f"Unsupported metric input type: {input_type}")
        self.input_type = input_type
        self.reset()

    def reset(self):
        self.total_intersection = 0
        self.total_union = 0
        self.total_gt_pixels = 0
        self.image_ious = []
        self.detected_targets = 0
        self.total_targets = 0
        self.false_alarm_pixels = 0
        self.total_pixels = 0

    def update(self, predictions, targets):
        """Update metrics with a batch of predictions and targets.

        Args:
            predictions: Tensor or numpy array shaped [B, 1, H, W] (logits or probabilities)
            targets: Tensor or numpy array shaped [B, 1, H, W] (binary ground truth {0, 1})
        """
        if isinstance(predictions, torch.Tensor):
            if self.input_type == "logits":
                predictions = torch.sigmoid(predictions)
            pred_np = (predictions.detach().cpu().numpy() > self.threshold)
        else:
            probabilities = np.asarray(predictions)
            if self.input_type == "logits":
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(probabilities, -80.0, 80.0)))
            pred_np = probabilities > self.threshold

        if isinstance(targets, torch.Tensor):
            target_np = (targets.detach().cpu().numpy() > 0.5)
        else:
            target_np = (targets > 0.5)

        if pred_np.shape != target_np.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {pred_np.shape} vs {target_np.shape}"
            )
        if pred_np.ndim != 4 or pred_np.shape[1] != 1:
            raise ValueError("Metrics expect tensors shaped [B, 1, H, W]")

        for prediction, target in zip(pred_np[:, 0], target_np[:, 0]):
            intersection = int(np.logical_and(prediction, target).sum())
            union = int(np.logical_or(prediction, target).sum())
            self.total_intersection += intersection
            self.total_union += union
            self.total_gt_pixels += int(target.sum())
            self.image_ious.append(
                float((intersection + self.eps) / (union + self.eps))
            )

            detected, target_count, false_pixels = self._object_statistics(
                prediction, target
            )
            self.detected_targets += detected
            self.total_targets += target_count
            self.false_alarm_pixels += false_pixels
            self.total_pixels += int(prediction.size)

    def _object_statistics(self, prediction, target):
        """Calculates detected targets via centroid matching and false alarm pixels."""
        pred_labels = measure.label(prediction.astype(np.uint8), connectivity=2)
        gt_labels = measure.label(target.astype(np.uint8), connectivity=2)
        pred_regions = measure.regionprops(pred_labels)
        gt_regions = measure.regionprops(gt_labels)
        unmatched = list(pred_regions)
        detected = 0
        matched_area = 0

        for gt_region in gt_regions:
            if not unmatched:
                break
            gt_centroid = np.asarray(gt_region.centroid, dtype=np.float64)
            for pred_position, pred_region in enumerate(unmatched):
                pred_centroid = np.asarray(pred_region.centroid, dtype=np.float64)
                # Centroid distance below 3 pixels criterion
                if np.linalg.norm(pred_centroid - gt_centroid) < self.match_distance:
                    detected += 1
                    matched_area += int(pred_region.area)
                    unmatched.pop(pred_position)
                    break

        false_pixels = int(prediction.sum()) - matched_area
        return detected, len(gt_regions), false_pixels

    def get(self):
        """Compute and return summary metric dictionary."""
        iou = self.total_intersection / (self.total_union + self.eps)
        niou = float(np.mean(self.image_ious)) if self.image_ious else 0.0
        pd = (
            self.detected_targets / self.total_targets if self.total_targets else 0.0
        )
        fa = (
            self.false_alarm_pixels / self.total_pixels if self.total_pixels else 0.0
        )
        # Fa in 10^-6 units as reported in manuscript tables
        fa_ppm = fa * 1e6

        return {
            "IoU": float(iou * 100.0),      # percentage
            "nIoU": float(niou * 100.0),    # percentage
            "Pd": float(pd * 100.0),        # percentage
            "Fa": float(fa_ppm),            # 10^-6 units
            "raw_iou": float(iou),
            "raw_niou": float(niou),
            "raw_pd": float(pd),
            "raw_fa": float(fa),
        }
