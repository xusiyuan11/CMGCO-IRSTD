"""Model builder factory."""

from pathlib import Path

from models.hcf import HCFnet
from models.msh import MSHNet
from models.msdanet import MSDANet
from models.dgnet import DGNet


def build_model(model_cfg):
    """Factory function to build detector models from dictionary config."""
    model_type = model_cfg.get("type", "HCFnet")
    mtype = model_type.lower()
    in_channels = model_cfg.get("in_channels", 3)
    out_channels = model_cfg.get("out_channels", 1)
    deep_supervision = model_cfg.get("deep_supervision", True)

    if mtype in ("hcfnet", "hcf"):
        base_channels = model_cfg.get("base_channels", 32)
        model = HCFnet(
            in_features=in_channels,
            out_features=out_channels,
            base_channels=base_channels,
            gt_ds=deep_supervision,
        )
    elif mtype in ("mshnet", "msh"):
        model = MSHNet(
            in_features=in_channels,
            out_features=out_channels,
            deep_supervision=deep_supervision,
        )
    elif mtype in ("msdanet", "msda", "msda_net"):
        model = MSDANet(
            in_channels=in_channels,
            out_channels=out_channels,
        )
    elif mtype in ("dgnet", "dg"):
        if in_channels != 1 or out_channels != 1:
            raise ValueError("The local DGNet protocol requires one input and one output channel")
        text_features = model_cfg.get("text_features")
        if text_features and not Path(text_features).is_absolute():
            text_features = Path(__file__).resolve().parents[1] / text_features
        model = DGNet(
            in_channels=in_channels,
            out_channels=out_channels,
            text_features_path=text_features,
        )
    elif mtype in ("samamba", "sam_mamba"):
        if in_channels != 3 or out_channels != 1:
            raise ValueError("SAMamba requires in_channels=3 and out_channels=1")
        checkpoint = model_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("SAMamba requires a local checkpoint path")
        if checkpoint and not Path(checkpoint).is_absolute():
            checkpoint = Path(__file__).resolve().parents[1] / checkpoint
        from models.samamba import SAMamba

        model = SAMamba(checkpoint_path=checkpoint)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

