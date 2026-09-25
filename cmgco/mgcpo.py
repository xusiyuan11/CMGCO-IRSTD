"""Paper-faithful Multi-Reward Group-Comparative Preference Optimization (MGCPO).

Strictly follows Section III-C of the CMGCO manuscript:
- Candidate generation: m_{i,g} = sigmoid(z_i + tau_s * eps_{i,g})
- Discrete segmentation rewards:
    r_IoU: overlap agreement
    r_Det: target discovery count
    r_FA: unmatched false alarm component count (negative)
    r_Frag: surplus fragmentation component count (negative)
- Within-group normalization: A_{i,g} = (R_{i,g} - mu_i) / (sigma_i + eps)
- Differentiable confidence score: mean negative entropy
- MGCPO loss: L_MGCPO = - (1 / (|B|*G)) * sum(sg[A_{i,g}] * s_theta(m_{i,g}))
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage


class SegmentationReward(nn.Module):
    """Computes the four discrete task-dependent rewards defined in Eq. 9."""

    def __init__(
        self,
        iou_weight=2.5,
        detection_weight=1.0,
        false_alarm_weight=1.0,
        fragmentation_weight=0.3,
        threshold=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.iou_weight = float(iou_weight)
        self.detection_weight = float(detection_weight)
        self.false_alarm_weight = float(false_alarm_weight)
        self.fragmentation_weight = float(fragmentation_weight)
        self.threshold = float(threshold)
        self.eps = float(eps)

    @torch.no_grad()
    def forward(self, candidates, targets):
        """Calculate four rewards for candidates shaped [B, G, 1, H, W].

        Args:
            candidates: Soft candidates in (0, 1), shaped [B, G, 1, H, W]
            targets: Binary ground truth {0, 1}, shaped [B, 1, H, W]
        """
        if candidates.ndim != 5 or candidates.shape[2] != 1:
            raise ValueError(f"candidates must have shape [B, G, 1, H, W], got {tuple(candidates.shape)}")
        if targets.ndim != 4 or targets.shape[1] != 1:
            raise ValueError(f"targets must have shape [B, 1, H, W], got {tuple(targets.shape)}")

        binary = candidates > self.threshold
        target_binary = targets > 0.5
        expanded_target = target_binary.unsqueeze(1)

        # 1. Overlap reward (r_IoU)
        intersection = (binary & expanded_target).sum(dim=(2, 3, 4)).float()
        union = (binary | expanded_target).sum(dim=(2, 3, 4)).float()
        r_iou = (intersection + self.eps) / (union + self.eps)

        binary_np = binary.detach().cpu().numpy().astype(np.uint8)
        target_np = target_binary.detach().cpu().numpy().astype(np.uint8)
        batch_size, group_size = binary_np.shape[:2]

        r_det = np.zeros((batch_size, group_size), dtype=np.float32)
        r_fa = np.zeros_like(r_det)
        r_frag = np.zeros_like(r_det)

        connectivity = np.ones((3, 3), dtype=np.uint8)
        for b in range(batch_size):
            gt_labels, gt_count = ndimage.label(target_np[b, 0], structure=connectivity)
            for g in range(group_size):
                pred_mask = binary_np[b, g, 0]
                pred_labels, pred_count = ndimage.label(pred_mask, structure=connectivity)

                # Count component intersections in one image pass. Label 0 is background.
                discovered_gt = np.unique(gt_labels[pred_mask.astype(bool)])
                detected = int(np.count_nonzero(discovered_gt))
                matched_pred = np.unique(pred_labels[target_np[b, 0].astype(bool)])
                false_alarms = int(pred_count - np.count_nonzero(matched_pred))

                # 4. Fragmentation penalty (r_Frag)
                fragmentation = max(0, pred_count - gt_count)

                r_det[b, g] = float(detected)
                r_fa[b, g] = -float(false_alarms)
                r_frag[b, g] = -float(fragmentation)

        device = candidates.device
        dtype = candidates.dtype
        r_det_tensor = torch.as_tensor(r_det, device=device, dtype=dtype)
        r_fa_tensor = torch.as_tensor(r_fa, device=device, dtype=dtype)
        r_frag_tensor = torch.as_tensor(r_frag, device=device, dtype=dtype)
        r_iou = r_iou.to(dtype=dtype)

        # Weighted composite reward R_{i,g}
        total_reward = (
            self.iou_weight * r_iou
            + self.detection_weight * r_det_tensor
            + self.false_alarm_weight * r_fa_tensor
            + self.fragmentation_weight * r_frag_tensor
        )

        return {
            "total": total_reward,
            "r_iou": r_iou,
            "r_det": r_det_tensor,
            "r_fa": r_fa_tensor,
            "r_frag": r_frag_tensor,
        }


class MGCPOObjective(nn.Module):
    """MGCPO objective module: samples candidate group, normalizes advantages, and applies confidence loss."""

    def __init__(
        self,
        group_size=16,
        temperature=1.0,
        iou_weight=2.5,
        detection_weight=1.0,
        false_alarm_weight=1.0,
        fragmentation_weight=0.3,
        threshold=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.group_size = int(group_size)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.reward_fn = SegmentationReward(
            iou_weight=iou_weight,
            detection_weight=detection_weight,
            false_alarm_weight=false_alarm_weight,
            fragmentation_weight=fragmentation_weight,
            threshold=threshold,
            eps=eps,
        )

    def forward(self, logits, targets):
        """Compute MGCPO loss and detailed reward statistics.

        Args:
            logits: Predicted raw logits before sigmoid [B, 1, H, W]
            targets: Binary ground truth mask [B, 1, H, W]
        """
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"logits must have shape [B, 1, H, W], got {tuple(logits.shape)}")

        candidates = []
        scores = []
        for _ in range(self.group_size):
            noise = torch.randn_like(logits)
            candidate = torch.sigmoid(logits + self.temperature * noise)
            candidates.append(candidate)

            # Candidate confidence score s_theta: mean negative entropy
            safe_c = candidate.clamp(self.eps, 1.0 - self.eps)
            score = (
                safe_c * torch.log(safe_c) + (1.0 - safe_c) * torch.log(1.0 - safe_c)
            ).mean(dim=(1, 2, 3))
            scores.append(score)

        candidate_tensor = torch.stack(candidates, dim=1)  # [B, G, 1, H, W]
        score_tensor = torch.stack(scores, dim=1)          # [B, G]

        # Compute discrete rewards on binarized candidates (detached)
        rewards = self.reward_fn(candidate_tensor.detach(), targets.detach())
        total_reward = rewards["total"]  # [B, G]

        # Within-group normalization (Eq. 10)
        group_mean = total_reward.mean(dim=1, keepdim=True)
        group_variance = ((total_reward - group_mean) ** 2).mean(dim=1, keepdim=True)
        group_std = torch.sqrt(group_variance)
        advantages = (total_reward - group_mean) / (group_std + self.eps)

        # MGCPO objective (Eq. 12): gradient flows ONLY through candidate score
        loss = -(advantages.detach() * score_tensor).mean()

        stats = {
            "mgcpo_loss": float(loss.detach().cpu()),
            "avg_reward": float(total_reward.mean().detach().cpu()),
            "r_iou": float(rewards["r_iou"].mean().detach().cpu()),
            "r_det": float(rewards["r_det"].mean().detach().cpu()),
            "r_fa": float(rewards["r_fa"].mean().detach().cpu()),
            "r_frag": float(rewards["r_frag"].mean().detach().cpu()),
        }
        return loss, stats

