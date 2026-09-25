from torch.utils.data import DataLoader
from .sirst_dataset import SIRSTDataset
from .transforms import get_train_transforms, get_test_transforms, get_eval_transforms


def build_dataset(dataset_cfg, mode="train"):
    """Build dataset from configuration dictionary."""
    data_root = dataset_cfg.get("data_root")
    idx_file = dataset_cfg.get("idx_file", None)
    img_size = dataset_cfg.get("img_size", 512)
    in_channels = dataset_cfg.get("in_channels", 3)

    dataset = SIRSTDataset(
        data_root=data_root,
        idx_file=idx_file,
        img_size=img_size,
        mode=mode,
        in_channels=in_channels,
    )
    return dataset


def build_dataloader(dataset, batch_size=4, shuffle=True, num_workers=4):
    """Build PyTorch DataLoader."""
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


__all__ = [
    "SIRSTDataset",
    "build_dataset",
    "build_dataloader",
    "get_train_transforms",
    "get_test_transforms",
    "get_eval_transforms",
]

