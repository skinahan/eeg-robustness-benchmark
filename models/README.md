# Models

EEG classification models for motor imagery, SSVEP, and P300 paradigms.

## Model Registry

Models are registered in `config.py` via `get_model_registry()`. The registry maps model names to factory functions that create skorch-compatible classifiers.

## Available Models

| Model | Description |
|-------|-------------|
| `eegnet` | EEGNet (Lawhern et al.) |
| `reegnet` | REEGNet |
| `cnn_ncp`, `cnn_ncp_v2` | CNN with NCP/LTC cells |
| `cnn_ncp_branch` | Branched CNN-NCP |
| `cnncfc_v2`, `cnncfc_compact` | CNN with CfC cells |
| `cnn_smallworld`, `cnn_wiredcfc_min` | Small-world and minimal WiredCfC |
| `cfc_only`, `ncp_only` | CfC-only and NCP-only baselines |
| `diva_ncp`, `branched_diva_ncp` | DIVA with NCP |
| `branched_lstm` | Branched LSTM |
| `branched_wiredcfc` | Branched WiredCfC (HYDRA-style) |
| `hydra_v2` | HYDRA v2 |
| `diva_full` | Full DIVA |
| `sppncp` | SPP-NCP |

Dynamic architectures (loaded from JSON):

- `wiredcfc_arch1`, `wiredcfc_arch2`, ... (from `outputs/architectures/`)
- `branched_wiredcfc_arch1`, ... (from `outputs/architectures/`)
- `hydra_v2_arch1`, ... (from `outputs/architectures/`)

## Adding a New Model

1. **Create the model module** in `models/` (e.g., `models/my_model.py`).

2. **Implement a factory function** that returns a skorch `NeuralNetClassifier` (or compatible):

   ```python
   def create_my_model_classifier(n_chans, n_times, n_outputs, **kwargs):
       return NeuralNetClassifier(MyModule(...), ...)
   ```

3. **Register in `config.py`**:

   - Add to `get_base_model_registry()`: `"my_model": create_my_model_classifier`
   - Or use `add_wiredcfc_architecture()`, `add_branched_wiredcfc_architecture()`, or `add_hydra_v2_architecture()` for JSON-based architectures.

4. **Add to `experiment_config.yaml`** if using experiment automation.

## Model Factory Signature

Factory functions typically accept:

- `n_chans`: Number of EEG channels
- `n_times`: Number of time samples
- `n_outputs`: Number of classes
- Additional kwargs (e.g., `drop_prob`, `lr`, `batch_size`) passed from the runner

## Dependencies

Models use `ncps` (Neural Circuit Policies) for CfC, LTC, and wiring classes. Install via `pip install ncps` or `conda env create -f environment.yml`.
