"""One paired baseline/CMGCO trainer with detector-specific source protocols."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cmgco.curriculum import ModelAwareCurriculum
from cmgco.losses import build_source_supervised_loss
from cmgco.mgcpo import MGCPOObjective
from data.dgnet_dataset import DGNetDataset
from data.sirst_dataset import SIRSTDataset
from models.builder import build_model
from utils.logger import MessageLogger, get_root_logger
from utils.metrics import PaperMetrics
from utils.misc import save_checkpoint, set_seed
from utils.yaml_utils import dict2str, parse_yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_NAMES = {"NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"}


def _model_type(options):
    return options["model"]["type"].lower()


def _main_logits(predictions, model_type, epoch=None, msh_warm_epochs=5):
    if not isinstance(predictions, (tuple, list)):
        return predictions
    if model_type in {"mshnet", "msh"} and epoch is not None and epoch <= msh_warm_epochs:
        return predictions[1]
    return predictions[0]


def _resolve_dataset_config(options, selected_dataset=None):
    data = options["dataset"]
    if selected_dataset is not None:
        if selected_dataset not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {selected_dataset}")
        for split, index_prefix in (("train", "train"), ("val", "test")):
            data[split]["data_root"] = str(PROJECT_ROOT / "datasets" / selected_dataset)
            data[split]["idx_file"] = f"img_idx/{index_prefix}_{selected_dataset}.txt"
    for split in ("train", "val"):
        root = Path(data[split]["data_root"])
        if not root.is_absolute():
            data[split]["data_root"] = str((PROJECT_ROOT / root).resolve())
    return Path(data["train"]["data_root"]).name


def _build_dataset(data_cfg, model_type, mode):
    if model_type in {"dgnet", "dg"}:
        return DGNetDataset(
            data_root=data_cfg["data_root"],
            img_size=data_cfg.get("img_size", 256),
            mode=mode,
        )
    return SIRSTDataset(
        data_root=data_cfg["data_root"],
        idx_file=data_cfg.get("idx_file"),
        img_size=data_cfg.get("img_size", 512),
        mode=mode,
        in_channels=data_cfg.get("in_channels", 3),
        preprocessing=data_cfg.get("preprocessing", model_type),
    )


@torch.no_grad()
def _evaluate(model, loader, device, model_type, epoch=None, msh_warm_epochs=5):
    was_training = model.training
    model.eval()
    metrics = PaperMetrics(input_type="logits")
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)
        logits = _main_logits(predictions, model_type, epoch, msh_warm_epochs)
        metrics.update(logits, masks)
    if was_training:
        model.train()
    return metrics.get()


def _readiness_evaluator(model, dataset, device, model_type, epoch, warm_epochs, threshold):
    @torch.no_grad()
    def evaluate(sample_ids):
        was_training = model.training
        model.eval()
        scores = {}
        for sid in sample_ids:
            image, mask, _ = dataset.get_eval_item(sid)
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device) > 0.5
            predictions = model(image)
            logits = _main_logits(predictions, model_type, epoch, warm_epochs)
            binary = torch.sigmoid(logits) > threshold
            intersection = (binary & mask).sum().float()
            union = (binary | mask).sum().float()
            scores[sid] = float((intersection + 1e-6) / (union + 1e-6))
        if was_training:
            model.train()
        return scores

    return evaluate


def _build_optimizer_and_scheduler(model, options, train_size):
    optim_cfg = options["optim"]
    train_cfg = options["train"]
    scheduler_cfg = options.get("scheduler", {})
    name = optim_cfg["type"].lower()
    lr = float(optim_cfg["lr"])
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_cfg['type']}")

    schedule = str(scheduler_cfg.get("type", "none")).lower()
    interval = str(scheduler_cfg.get("step_interval", "epoch")).lower()
    if schedule in {"none", "null"}:
        scheduler = None
    elif schedule == "cosineannealinglr":
        if interval == "iter":
            steps_per_epoch = math.ceil(train_size / int(train_cfg["batch_size"]))
            total_steps = int(train_cfg["epochs"]) * steps_per_epoch
        elif interval == "epoch":
            total_steps = int(train_cfg["epochs"])
        else:
            raise ValueError(f"Unsupported scheduler step interval: {interval}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=float(scheduler_cfg.get("eta_min", 0.0))
        )
    elif schedule == "multisteplr":
        if interval != "epoch":
            raise ValueError("DGNet MultiStepLR must step each epoch")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(scheduler_cfg["milestones"]),
            gamma=float(scheduler_cfg["gamma"]),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg.get('type')}")
    return optimizer, scheduler, interval


def _build_dgnet_auxiliary(model_type, options, dataset, device):
    if model_type not in {"dgnet", "dg"}:
        return None
    cda_cfg = options.get("loss", {}).get("cda", {})
    if not cda_cfg.get("enabled", False):
        return None
    from cmgco.dgnet_cda import ConsensusKnowledgeDirectionalAlignmentLoss

    weight = Path(cda_cfg["checkpoint"])
    if not weight.is_absolute():
        weight = PROJECT_ROOT / weight
    if not weight.is_file():
        raise FileNotFoundError(f"DGNet CLIP checkpoint not found: {weight}")
    statistics = dataset.source.img_norm_cfg
    return ConsensusKnowledgeDirectionalAlignmentLoss(
        device=device,
        ratio=float(cda_cfg.get("ratio", 0.8)),
        model_name=str(weight),
        image_mean=statistics["mean"],
        image_std=statistics["std"],
    ).to(device)


def _dgnet_cda_weight(epoch):
    zero_based_epoch = epoch - 1
    if zero_based_epoch < 5:
        return 0.0
    if zero_based_epoch < 300:
        return 1.0
    if zero_based_epoch < 400:
        return 0.5
    if zero_based_epoch < 500:
        return 0.25
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Unified paired CMGCO trainer")
    parser.add_argument("--opt", required=True, help="Detector configuration YAML")
    parser.add_argument("--dataset", choices=sorted(DATASET_NAMES), help="Override dataset in YAML")
    parser.add_argument("--variant", choices=("baseline", "cmgco"), help="Paired training variant")
    args = parser.parse_args()

    options = parse_yaml(args.opt)
    model_type = _model_type(options)
    dataset_name = _resolve_dataset_config(options, args.dataset)
    curriculum_cfg = options.get("curriculum", {})
    mgcpo_cfg = options.get("mgcpo", {})
    if args.variant is not None:
        curriculum_cfg["enabled"] = args.variant == "cmgco"
        mgcpo_cfg["enabled"] = args.variant == "cmgco"
    variant = args.variant or ("cmgco" if curriculum_cfg.get("enabled") and mgcpo_cfg.get("enabled") else "custom")

    exp_cfg = options.get("exp", {})
    save_root = Path(exp_cfg.get("save_dir", PROJECT_ROOT / "experiments" / model_type))
    if not save_root.is_absolute():
        save_root = PROJECT_ROOT / save_root
    save_dir = save_root / dataset_name / variant
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger(log_file=str(save_dir / "train.log"))
    logger.info("Loaded configuration:\n" + dict2str(options))
    logger.info(f"Pair: {model_type} / {dataset_name} / {variant}")

    set_seed(exp_cfg.get("seed", 42))
    device = torch.device(f"cuda:{exp_cfg.get('device', 0)}" if torch.cuda.is_available() else "cpu")
    model = build_model(options["model"]).to(device)
    train_dataset = _build_dataset(options["dataset"]["train"], model_type, "train")
    val_dataset = _build_dataset(options["dataset"]["val"], model_type, "val")
    train_cfg = options["train"]
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=int(train_cfg.get("val_batch_size", 1)), shuffle=False, num_workers=workers)

    loss_cfg = options.get("loss", {})
    supervised = build_source_supervised_loss(model_type, loss_cfg).to(device)
    warm_epochs = int(loss_cfg.get("warm_epochs", 5))
    cda = _build_dgnet_auxiliary(model_type, options, train_dataset, device)
    optimizer, scheduler, scheduler_interval = _build_optimizer_and_scheduler(model, options, len(train_dataset))
    curriculum = None
    if curriculum_cfg.get("enabled", False):
        curriculum = ModelAwareCurriculum(
            all_ids=train_dataset.all_ids,
            bootstrap_epochs=curriculum_cfg.get("bootstrap_epochs", 20),
            readiness_threshold=curriculum_cfg.get("readiness_threshold", 0.5),
            update_interval=curriculum_cfg.get("update_interval", 5),
            force_all_epoch=curriculum_cfg.get("force_all_epoch", 220),
        )
    mgcpo = None
    if mgcpo_cfg.get("enabled", False):
        mgcpo = MGCPOObjective(
            group_size=mgcpo_cfg.get("group_size", 16),
            temperature=mgcpo_cfg.get("temperature", 1.0),
            iou_weight=mgcpo_cfg.get("iou_weight", 2.5),
            detection_weight=mgcpo_cfg.get("detection_weight", 1.0),
            false_alarm_weight=mgcpo_cfg.get("false_alarm_weight", 1.0),
            fragmentation_weight=mgcpo_cfg.get("fragmentation_weight", 0.3),
            threshold=mgcpo_cfg.get("threshold", 0.5),
        )

    epochs = int(train_cfg["epochs"])
    message_logger = MessageLogger(f"{model_type}/{dataset_name}/{variant}", epochs, logger)
    best_iou = float("-inf")
    for epoch in range(1, epochs + 1):
        curriculum_info = ""
        if curriculum is not None:
            score_fn = _readiness_evaluator(
                model, train_dataset, device, model_type, epoch, warm_epochs,
                curriculum_cfg.get("readiness_threshold", 0.5),
            )
            status = curriculum.maybe_update(epoch, score_fn)
            if status["updated"]:
                train_dataset.set_active_ids(curriculum.active_ids)
            curriculum_info = f"{status['phase']}: {status['active_count']} active"
        model.train()
        train_metrics = PaperMetrics(input_type="logits")
        loss_sum = 0.0
        actual_steps = 0
        for batch in train_loader:
            if model_type in {"dgnet", "dg"}:
                original, augmented, masks, _ = batch
                images = torch.cat((original, augmented), dim=0).to(device)
                masks = torch.cat((masks, masks), dim=0).to(device)
            else:
                images, masks, _ = batch
                images = images.to(device)
                masks = masks.to(device)
            predictions = model(images)
            logits = _main_logits(predictions, model_type, epoch, warm_epochs)
            supervised_loss = supervised(predictions, masks, epoch)
            total = supervised_loss
            if cda is not None and _dgnet_cda_weight(epoch) > 0.0:
                total = total + _dgnet_cda_weight(epoch) * cda(torch.sigmoid(logits), masks, images)
            if mgcpo is not None and epoch > int(mgcpo_cfg.get("activation_epoch", 20)):
                preference_loss, _ = mgcpo(logits, masks)
                total = total + float(mgcpo_cfg.get("weight", 1.0)) * preference_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            if scheduler is not None and scheduler_interval == "iter":
                scheduler.step()
            loss_sum += float(total.detach())
            actual_steps += 1
            train_metrics.update(logits.detach(), masks)
        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()
        if actual_steps == 0:
            raise RuntimeError(f"Empty active training set at epoch {epoch}")

        val_result = None
        if epoch % int(train_cfg.get("val_interval", 1)) == 0:
            val_result = _evaluate(
                model, val_loader, device, model_type, epoch, warm_epochs,
            )
            if val_result["IoU"] > best_iou:
                best_iou = val_result["IoU"]
                save_checkpoint(model, optimizer, scheduler, epoch, best_iou, str(save_dir / "best_model.pth"))
        message_logger.log_epoch(
            epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            train_loss=loss_sum / actual_steps,
            train_metrics=train_metrics.get(),
            val_metrics=val_result,
            extra_info=curriculum_info,
        )
        if epoch % int(train_cfg.get("save_interval", 50)) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, str(save_dir / f"epoch_{epoch}.pth"))

    save_checkpoint(model, optimizer, scheduler, epochs, best_iou, str(save_dir / "latest_model.pth"))
    logger.info(f"Training completed. Best IoU: {best_iou:.2f}%")


if __name__ == "__main__":
    main()
