# EEG Noise Robustness Experiments

A unified framework for evaluating EEG classification models under various noise conditions, built on [MOABB](https://neurotechx.github.io/moabb/) (Mother of All BSS Benchmarks).

## Overview

This repository provides:

- **Unified experiment runner** (`evaluation/unified_experiment_runner.py`) for training and evaluating EEG models
- **Multiple evaluation modes**: WithinSession, CrossSession, CrossSubject
- **Robustness evaluation** (`test_perturb` mode): clean training, test-time corruption with Gaussian, dropout, EOG, and spike noise
- **Architecture refinement**: NAS and topology optimization for WiredCfC/HYDRA models (see `architecture_refinement/`)
- **Experiment automation**: Identify missing experiments and generate job scripts

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd moabb_experiments

# Create conda environment
conda env create -f environment.yml
conda activate ncp_robustness_proj
```

**Dependencies**: The project uses `ncps` (Neural Circuit Policies) as an external package. It is included in `environment.yml`. Install via:

```bash
pip install ncps  # or use conda env create -f environment.yml
```

### 2. Download Datasets

```bash
python download_datasets.py
```

### 3. Run an Experiment

**Robustness evaluation (test_perturb):**
```bash
python evaluation/unified_experiment_runner.py \
  --model eegnet \
  --dataset BNCI2014_001 \
  --mode test_perturb \
  --subjects 1 2 3 \
  --eval_mode CrossSession \
  --seed 42 \
  --overwrite
```

**Multirun (single noise type/intensity):**
```bash
python evaluation/unified_experiment_runner.py \
  --model eegnet \
  --dataset BNCI2014_001 \
  --mode multirun \
  --subjects 1 \
  --eval_mode CrossSession \
  --noise_type gaussian \
  --intensity 10.0 \
  --seed 42
```

### 4. Experiment Automation

To identify missing experiments and generate run scripts:

```bash
python experiment_automation.py --config experiment_config.yaml --local
```

See [README_automation.md](README_automation.md) for details.

## Project Structure

```
moabb_experiments/
├── evaluation/           # Core experiment runner, metrics, utilities
├── models/               # EEG classification models (EEGNet, REEGNet, HYDRA, etc.)
├── augmentation/         # Noise augmentation (Gaussian, dropout, EOG)
├── architecture_refinement/  # NAS, topology optimization, Paper3 experiments
├── ablations/            # Ablation study scripts
├── analysis/             # Result analysis and plotting
├── config.py             # Model registry, paradigm/dataset configuration
├── experiment_config.yaml    # Experiment automation configuration
└── docs/                 # Additional documentation
```

## Models

Available models (see `config.py` and `models/`):

- **EEGNet**, **REEGNet**: Standard EEG architectures
- **CNN-NCP**, **CNN-CfC**: CNN with Neural Circuit Policy / Closed-form Continuous-time cells
- **HYDRA**, **BranchedWiredCfC**: Multi-branch architectures with optimized wirings
- **DIVA-NCP**, **DIVA-Full**: Divergent architecture variants

Architectures can be loaded from `outputs/architectures/*.json` and `.model_registry/`.

## Documentation

- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) – Command cheat sheet
- [README_automation.md](README_automation.md) – Experiment automation
- [BI2015A_P300_CONFIG.md](BI2015A_P300_CONFIG.md) – P300 dataset configuration
- [docs/](docs/) – Implementation notes (chunked training, CrossSubject, etc.)
- [architecture_refinement/README.md](architecture_refinement/README.md) – NAS and topology optimization

## License

See [LICENSE](LICENSE) for details.
