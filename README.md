# CMGCO for Infrared Small-Target Detection

**Curriculum-Guided Multi-Reward Group-Comparative Optimization (CMGCO)** is a training framework for infrared small-target detection. It combines model-aware curriculum learning with Multi-Reward Group-Comparative Preference Optimization (MGCPO) and integrates both components into detector training.

This repository contains model implementations for **HCFNet, MSHNet, MSDANet, DGNet, and SAMamba**. All five detectors use the shared training and evaluation interface with paired `baseline` and `cmgco` variants. SAMamba's network and SAM2 backbone components remain under `basicseg/` and `sam2_configs/`; `models/samamba/` exposes that network to the unified model builder without duplicating its architecture.

## Repository structure

```text
CMGCO-IRSTD/
├── cmgco/                         # Core training framework
│   ├── curriculum.py             # Sample readiness and curriculum scheduling
│   ├── mgcpo.py                  # Candidate generation and MGCPO objective
│   ├── losses.py                 # Detector-specific supervised objectives
│   └── dgnet_cda.py              # DGNet auxiliary alignment loss
├── models/                        # Detectors used by the unified interface
│   ├── builder.py                # Model construction from configurations
│   ├── hcf/                      # HCFNet
│   ├── msh/                      # MSHNet
│   ├── msdanet/                  # MSDANet
│   ├── dgnet/                    # DGNet
│   ├── samamba/                  # SAMamba adapter to the source network
│   └── common/                   # Shared network modules
├── configs/                       # Detector and training configurations
├── data/                          # Dataset loaders and preprocessing
├── utils/                         # Metrics, checkpoints, logging, and configuration I/O
├── datasets/                      # Versioned train/test split lists only
│   ├── NUAA-SIRST/img_idx/
│   ├── NUDT-SIRST/img_idx/
│   └── IRSTD-1K/img_idx/
├── basicseg/                      # SAMamba and SAM2 source components
│   └── networks/
│       ├── SAMamba.py             # SAMamba detector
│       └── sam2/                 # SAM2 backbone and supporting modules
├── sam2_configs/                  # SAM2 architecture configurations
├── train.py                       # Unified training entry point
├── train_samamba.py               # SAMamba alias for the unified trainer
├── test.py                        # Checkpoint evaluation and prediction export
├── train_*.py                     # Other detector aliases for train.py
├── .gitignore                     # Local data, weights, and output exclusions
├── requirements.txt
├── requirements-samamba.txt       # Additional SAMamba dependencies
└── README.md
```

The tree above shows versioned repository content. Dataset images and masks, pretrained weights, checkpoints, logs, and analysis outputs are prepared or generated locally and excluded from Git.

## Installation

Clone the repository and create a Python 3.10 environment:

```bash
git clone https://github.com/xusiyuan11/CMGCO-IRSTD.git
cd CMGCO-IRSTD
conda create -n cmgco python=3.10 pip -y
conda activate cmgco
```

Install a matching PyTorch and torchvision build using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). For GPU training, select a CUDA build compatible with your NVIDIA driver. Then install the project dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The dependency specification includes PyTorch 2.3 or later, Albumentations 1.3.1, and a pinned revision of OpenAI CLIP. Git is required to install the CLIP dependency. SAMamba additionally requires the dependencies in `requirements-samamba.txt`, including a CUDA-compatible `mamba-ssm` build. Install them in a Linux or WSL CUDA environment after PyTorch. Run the Python commands below from the repository root.

## Data preparation

Prepare each dataset under `datasets/<dataset>/` with `images/` and `masks/` directories alongside the versioned `img_idx/` split lists. Each split file contains one image identifier per line; images and masks share the same filename stem.

| Dataset | Training images | Test images | Split directory |
| --- | ---: | ---: | --- |
| NUAA-SIRST | 213 | 214 | [datasets/NUAA-SIRST/img_idx](datasets/NUAA-SIRST/img_idx) |
| NUDT-SIRST | 663 | 664 | [datasets/NUDT-SIRST/img_idx](datasets/NUDT-SIRST/img_idx) |
| IRSTD-1K | 800 | 201 | [datasets/IRSTD-1K/img_idx](datasets/IRSTD-1K/img_idx) |

Split files follow the names `train_<dataset>.txt` and `test_<dataset>.txt`.

### DGNet auxiliary assets

Prepare the following files before running DGNet:

| File | Purpose |
| --- | --- |
| `weights/dgnet/dgnet_text_features.pth` | Fixed target and background text features used during training and evaluation |
| `weights/dgnet/ViT-B-32.pt` | CLIP checkpoint used by the auxiliary CDA loss during training |

Their paths are configured in [configs/dgnet_sirst.yaml](configs/dgnet_sirst.yaml).

### SAMamba backbone asset

Place the SAM2 Hiera-S checkpoint at `weights/samamba/sam2_hiera_small.pt`, as configured in [configs/samamba_sirst.yaml](configs/samamba_sirst.yaml). The file remains local and is excluded from Git.

## Models and configurations

Input sizes and default training settings for the five detectors are summarized below.

| Detector | Configuration / model source | Input size | Epochs | Batch size |
| --- | --- | --- | ---: | ---: |
| HCFNet | [hcfnet_sirst.yaml](configs/hcfnet_sirst.yaml) | 3-channel, 512 × 512 | 300 | 4 |
| MSHNet | [mshnet_sirst.yaml](configs/mshnet_sirst.yaml) | RGB, 512 × 512 | 400 | 4 |
| MSDANet | [msdanet_sirst.yaml](configs/msdanet_sirst.yaml) | RGB, 512 × 512 | 500 | 4 |
| DGNet | [dgnet_sirst.yaml](configs/dgnet_sirst.yaml) | Grayscale, 256 × 256 | 600 | 16 |
| SAMamba | [samamba_sirst.yaml](configs/samamba_sirst.yaml) | 3-channel, 1024 × 1024 | 400 | 2 |

SAMamba's training and test pipelines both use **1024 × 1024** OpenCV BGR inputs. Its SAM2 backbone configuration is [sam2_hiera_s.yaml](sam2_configs/sam2_hiera_s.yaml). The model is constructed through `models/samamba/` and uses the same `train.py` implementation as the other detectors.

Each YAML file defines the model, dataset preprocessing, supervised objective, optimizer, learning-rate schedule, and CMGCO settings. Common settings are organized as follows:

| Configuration section | Settings |
| --- | --- |
| `exp` | Experiment name, output directory, device, and random seed |
| `model` | Detector architecture and auxiliary asset paths |
| `dataset` | Dataset paths, split files, channels, and input resolution |
| `train` | Batch sizes, epochs, data-loader workers, and save intervals |
| `optim` / `scheduler` | Optimizer and learning-rate schedule |
| `loss` | Detector-specific supervision |
| `curriculum` | Bootstrap, readiness assessment, and full-data refinement |
| `mgcpo` | Activation epoch, candidate sampling, reward weights, and loss weight |

## Training

Select a detector with `--opt`, a dataset with `--dataset`, and the training variant with `--variant`.

| Variant | Training setup |
| --- | --- |
| `baseline` | Detector-specific supervised training |
| `cmgco` | Supervised training with curriculum learning and MGCPO |

For example, train an HCFNet baseline and its CMGCO counterpart on NUDT-SIRST:

```bash
python train.py --opt configs/hcfnet_sirst.yaml --dataset NUDT-SIRST --variant baseline
python train.py --opt configs/hcfnet_sirst.yaml --dataset NUDT-SIRST --variant cmgco
```

Use another configuration from the table above to change the detector. Available datasets are `NUAA-SIRST`, `NUDT-SIRST`, and `IRSTD-1K`. The `--dataset` option selects the dataset directory and its split lists.

For SAMamba in a Linux or WSL CUDA environment:

```bash
python -m pip install -r requirements-samamba.txt
python train_samamba.py --opt configs/samamba_sirst.yaml --dataset NUAA-SIRST --variant baseline
python train_samamba.py --opt configs/samamba_sirst.yaml --dataset NUAA-SIRST --variant cmgco
```

Training writes logs and checkpoints under `exp.save_dir/<dataset>/<variant>/`; these outputs are not tracked by Git. `best_model.pth` is selected by validation IoU, while `latest_model.pth` records the final epoch.

The `train_hcfnet.py`, `train_mshnet.py`, `train_msdanet.py`, `train_dgnet.py`, and `train_samamba.py` entry points accept the same arguments as `train.py`.

## Evaluation

Evaluate a checkpoint with the corresponding detector configuration and dataset:

```bash
python test.py --opt configs/hcfnet_sirst.yaml --dataset NUDT-SIRST --ckpt experiments/CMGCO_HCFNet_SIRST/NUDT-SIRST/cmgco/best_model.pth
```

The command reports IoU, nIoU, P<sub>d</sub>, and F<sub>a</sub>. Add `--save-preds` to export binary prediction masks:

```bash
python test.py --opt configs/hcfnet_sirst.yaml --dataset NUDT-SIRST --ckpt experiments/CMGCO_HCFNet_SIRST/NUDT-SIRST/cmgco/best_model.pth --save-preds results/hcfnet_nudt
```

For SAMamba, use its configuration and the checkpoint from the matching run:

```bash
python test.py --opt configs/samamba_sirst.yaml --dataset NUAA-SIRST --ckpt experiments/CMGCO_SAMamba_SIRST/NUAA-SIRST/cmgco/best_model.pth
```

## Acknowledgements

This work builds on the detector implementations of HCFNet, MSHNet, MSDA-Net, DGNet, and SAMamba. The SAMamba integration uses SAM2 backbone components, while DGNet uses OpenAI CLIP for auxiliary supervision. We thank the authors and maintainers of these projects for making their work available.
