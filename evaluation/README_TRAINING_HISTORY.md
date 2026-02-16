# Training History Logging

## Overview

The unified experiment runner now automatically logs model training history after each `.fit()` call. This enables post-hoc analysis of training dynamics, helping to detect overfitting and underfitting patterns.

## What Gets Logged

After each model training session, the system saves:
- **Epoch-by-epoch metrics**: Loss values, accuracy, and other metrics tracked during training
- **Training metadata**: Subject ID, session, fold index, and experiment mode
- **File format**: JSON files for easy parsing and visualization

## File Location

Training history files are saved to:
```
results/{paradigm}/{dataset}/{model}/{eval_mode}/{seed}/sub-{subject}/{session}/training_history/
```

Example:
```
results/MotorImagery/BNCI2014_001/eegnet/CrossSessionEvaluation/42/sub-001/0train/training_history/
```

## File Naming Convention

Files are named using the pattern:
```
history_sub{subject:03d}_sess{session}_fold{fold}_{mode}.json
```

Examples:
- `history_sub001_sess0train_fold0_test_perturb.json` - Training without hyperparameter tuning
- `history_sub001_sess0train_fold0_test_perturb_tuned.json` - Training after hyperparameter optimization
- `history_sub001_sess0train_fold0_test_perturb_retrained.json` - Re-training without early stopping

## When History is Logged

### Regular Training (No Tuning)
History is logged immediately after each `.fit()` call:
- After training in `_evaluate_without_tuning()`
- After training in `_train_and_evaluate_perturb()`

### Hyperparameter Tuning
When `--tune` flag is used, history is logged **only after the parameter search completes**:
- The final model is trained with optimal parameters
- History from the final training is saved (not from intermediate tuning trials)
- Filename includes `_tuned` suffix to distinguish from non-tuned runs

### Re-training
If a model is re-trained (e.g., without early stopping due to underfitting):
- Both the initial training and re-training histories are saved
- Re-trained models include `_retrained` suffix in filename

## Example History File Structure

```json
[
  {
    "epoch": 1,
    "train_loss": 0.693,
    "valid_loss": 0.689,
    "train_acc": 0.512,
    "valid_acc": 0.525,
    "dur": 2.34
  },
  {
    "epoch": 2,
    "train_loss": 0.671,
    "valid_loss": 0.665,
    "train_acc": 0.587,
    "valid_acc": 0.592,
    "dur": 2.28
  },
  ...
]
```

## Usage for Analysis

### Load and Plot Loss Curves

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('history_sub001_sess0train_fold0_test_perturb.json', 'r') as f:
    history = json.load(f)

# Extract metrics
epochs = [h['epoch'] for h in history]
train_loss = [h['train_loss'] for h in history]
valid_loss = [h['valid_loss'] for h in history]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, valid_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.show()
```

### Detect Overfitting

```python
# Calculate gap between train and validation loss
final_epoch = history[-1]
loss_gap = final_epoch['valid_loss'] - final_epoch['train_loss']

if loss_gap > 0.1:  # Threshold depends on your domain
    print(f"Potential overfitting detected (gap: {loss_gap:.3f})")
```

### Detect Underfitting

```python
# Check if validation loss is still decreasing
last_5_epochs = history[-5:]
valid_losses = [h['valid_loss'] for h in last_5_epochs]

# Simple check: is there a decreasing trend?
if valid_losses[-1] < valid_losses[0]:
    print("Model may benefit from more training epochs")
```

## Integration with Existing Workflow

The history logging is fully integrated and requires no changes to existing scripts:

```bash
# Runs as before, but now also saves training history
python evaluation/unified_experiment_runner.py \
    --model eegnet \
    --dataset BNCI2014_001 \
    --subjects 1 2 3 \
    --mode test_perturb \
    --eval_mode CrossSession \
    --seed 42 \
    --tune
```

## Implementation Details

### Function: `save_training_history()`

Located in `unified_experiment_runner.py`, this function:
1. Extracts history from the trained model's `.history` attribute
2. Converts PyTorch tensors and NumPy arrays to JSON-serializable format
3. Saves to appropriately named file in the training_history subdirectory
4. Handles errors gracefully with warnings if history cannot be saved

### Key Points

- **No performance impact**: History saving is fast and happens after training completes
- **Backward compatible**: Existing results are not affected
- **Optional analysis**: History files are separate from main results
- **Tuning-aware**: During hyperparameter search, only the final model's history is saved (not intermediate trials)

## Future Enhancements

Potential additions:
- Automatic overfitting/underfitting detection in the training loop
- Learning rate schedule tracking
- Gradient statistics
- Early stopping trigger information
- Visualization scripts in `analysis/` directory

