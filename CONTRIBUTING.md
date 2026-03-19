# Contributing

Thank you for your interest in contributing to the EEG noise robustness experiments project.

## Development Setup

1. Clone the repository and create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ncp_robustness_proj
   ```

2. Install in development mode (if applicable):
   ```bash
   pip install -e .
   ```

## Code Style

- Use Python 3.10+
- Follow PEP 8
- Add docstrings to public functions and classes
- Use type hints where practical

## Adding a New Model

1. Create a model module in `models/` (see [models/README.md](models/README.md))
2. Implement a factory function returning a skorch-compatible classifier
3. Register in `config.py` via `get_base_model_registry()` or the appropriate architecture registry
4. Add to `experiment_config.yaml` if using experiment automation

## Running Tests

```bash
python evaluation/unified_experiment_runner.py --help
python experiment_automation.py --help
```

For a minimal smoke test:
```bash
python evaluation/unified_experiment_runner.py \
  --model eegnet --dataset BNCI2014_001 --mode test_perturb \
  --subjects 1 --eval_mode CrossSession --seed 999 --overwrite \
  --test_perturb_gaussian_only --test_perturb_gaussian_alpha_grid 0,0.5,1
```

## Project Structure

- `evaluation/` – Experiment runner and utilities
- `models/` – EEG classification models
- `augmentation/` – Noise augmentation
- `architecture_refinement/` – NAS and topology optimization
- `config.py` – Model registry and configuration

## Questions

Open an issue for questions or suggestions.
