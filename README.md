# MOABB EEG Model Benchmarking

This repository provides a clean, modular framework for evaluating deep learning models on EEG motor imagery classification tasks using [MOABB](https://github.com/NeuroTechX/moabb).

## Features
- EEGNet, REEGNet baseline models (PyTorch + Skorch)
- Integration with MOABB evaluation pipelines
- Support for data augmentation: dropout, Gaussian noise, simulated EOG
- Reproducible experiment configuration

## Directory Structure
```
moabb_experiments/
├── models/
│   ├── eegnet.py
│   └── reegnet.py
├── augmentation/
│   └── noise.py
├── evaluation/
│   └── run_experiment.py
├── config.py
└── results/
```

## Getting Started
```bash
# Clone the repo and install dependencies
pip install -r requirements.txt

# Example: run EEGNet baseline
python evaluation/run_experiment.py --model eegnet

# Example: run REEGNet with EOG augmentation
python evaluation/run_experiment.py --model reegnet --noise_type eog --intensity 4.0
```

## Requirements
- torch
- braindecode
- moabb
- scikit-learn
- mne
- numpy

## License
MIT License
