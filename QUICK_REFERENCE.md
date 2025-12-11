# Training History Logging - Quick Reference

## Changes at a Glance

### Modified File: `evaluation/unified_experiment_runner.py`

#### 1. Added Import (Line 49)
```python
import json
```

#### 2. New Function: `save_training_history()` (Lines 65-134)
```python
def save_training_history(model, output_path: str, fold_idx: int = None, ...):
    """Save training history from fitted model to JSON file."""
    # Extracts model.history
    # Converts to JSON-serializable format
    # Saves to training_history/ subdirectory
```

#### 3. New Method: `_get_history_output_path()` (Lines 234-245)
```python
def _get_history_output_path(self):
    """Get the base output path for saving training history."""
    # Returns appropriate directory for history files
```

#### 4. History Logging After Training (6 locations)

**Location 1**: After hyperparameter tuning (Lines 474-483)
```python
final_model.fit(X_train, y_train)
training_time = time.time() - start_time

# ✅ NEW: Save training history after hyperparameter tuning is complete
output_path = self._get_history_output_path()
save_training_history(
    final_model, 
    output_path, 
    fold_idx=fold_idx, 
    subject=self.current_subject,
    session=str(self.current_session),
    mode=f"{self.mode}_tuned"
)
```

**Location 2**: After re-training (tuned) (Lines 512-521)
```python
final_model.fit(X_train, y_train)
training_time = time.time() - start_time

# ✅ NEW: Save re-training history
output_path = self._get_history_output_path()
save_training_history(
    final_model, 
    output_path, 
    fold_idx=fold_idx, 
    subject=self.current_subject,
    session=str(session),
    mode=f"{self.mode}_tuned_retrained"
)
```

**Location 3**: After training without tuning (Lines 572-581)
```python
model.fit(X_train, y_train)

# ✅ NEW: Save training history
output_path = self._get_history_output_path()
save_training_history(
    model, 
    output_path, 
    fold_idx=fold_idx, 
    subject=self.current_subject,
    session=str(self.current_session),
    mode=self.mode
)
```

**Location 4**: After training in test_perturb mode (Lines 689-698)
```python
model.fit(X_train, y_train)
training_time = time.time() - start_time

# ✅ NEW: Save training history
output_path = self._get_history_output_path()
save_training_history(
    model, 
    output_path, 
    fold_idx=fold_idx, 
    subject=self.current_subject,
    session=str(session),
    mode=self.mode
)
```

**Location 5**: After re-training in test_perturb mode (Lines 726-735)
```python
model.fit(X_train, y_train)
training_time = time.time() - start_time

# ✅ NEW: Save re-training history
output_path = self._get_history_output_path()
save_training_history(
    model, 
    output_path, 
    fold_idx=fold_idx, 
    subject=self.current_subject,
    session=str(session),
    mode=f"{self.mode}_retrained"
)
```

## New Files Created

### 1. Documentation
- `evaluation/README_TRAINING_HISTORY.md` - Complete feature documentation
- `TRAINING_HISTORY_UPDATE.md` - Feature overview and benefits
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

### 2. Analysis Tools
- `analysis/analyze_training_history.py` - Command-line analysis tool
- `analysis/example_training_history_analysis.py` - Usage examples

### 3. Reference
- `QUICK_REFERENCE.md` - This file

## Output Structure

After running an experiment, you'll see:

```
results/MotorImagery/BNCI2014_001/eegnet/CrossSessionEvaluation/42/sub-001/0train/
├── training_history/                    # ✅ NEW DIRECTORY
│   ├── history_sub001_sess0train_fold0_test_perturb.json
│   ├── history_sub001_sess0train_fold1_test_perturb.json
│   ├── history_sub001_sess0train_fold0_test_perturb_tuned.json
│   └── history_sub001_sess0train_fold0_test_perturb_retrained.json
├── eegnet_test_perturb_gaussian_10.0_subject_001_seed42.csv
└── [other result files]
```

## Example History File

```json
[
  {
    "epoch": 1,
    "train_loss": 0.6931,
    "valid_loss": 0.6890,
    "train_acc": 0.5120,
    "valid_acc": 0.5250,
    "dur": 2.34
  },
  {
    "epoch": 2,
    "train_loss": 0.6712,
    "valid_loss": 0.6652,
    "train_acc": 0.5870,
    "valid_acc": 0.5920,
    "dur": 2.28
  }
]
```

## Usage Commands

### Run Experiment (History Logged Automatically)
```bash
python evaluation/unified_experiment_runner.py \
    --model eegnet \
    --dataset BNCI2014_001 \
    --subjects 1 \
    --mode test_perturb \
    --eval_mode CrossSession \
    --seed 42 \
    --tune
```

### Analyze History
```bash
# Single file
python analysis/analyze_training_history.py \
    --history_file results/.../history_sub001_sess0train_fold0_test_perturb.json

# Multiple files comparison
python analysis/analyze_training_history.py \
    --history_dir results/.../training_history/ \
    --compare \
    --output_dir outputs/plots
```

### Quick Python Analysis
```python
import json
import matplotlib.pyplot as plt

# Load
with open('history_sub001_sess0train_fold0_test_perturb.json') as f:
    history = json.load(f)

# Check overfitting
final = history[-1]
gap = final['valid_loss'] - final['train_loss']
print(f"Train/Valid Gap: {gap:.4f}")
if gap > 0.1:
    print("⚠️ Overfitting detected!")

# Plot
epochs = [h['epoch'] for h in history]
plt.plot(epochs, [h['train_loss'] for h in history], label='Train')
plt.plot(epochs, [h['valid_loss'] for h in history], label='Valid')
plt.legend()
plt.show()
```

## Key Features

✅ **Automatic**: No code changes needed in your scripts  
✅ **Tuning-Aware**: Only logs after parameter search completes  
✅ **Robust**: Handles all data types and edge cases  
✅ **Comprehensive**: Saves all epoch metrics  
✅ **Lightweight**: Small JSON files (~10-100 KB)  
✅ **Compatible**: Works with all models and modes  

## Important Notes

1. **During hyperparameter tuning**: History is saved **only after** the parameter search completes (not during intermediate trials)
2. **Re-training**: When models are re-trained (e.g., without early stopping), both histories are saved with different suffixes
3. **File naming**: Suffix indicates the training type:
   - No suffix: Regular training
   - `_tuned`: After hyperparameter optimization
   - `_retrained`: After re-training without early stopping
   - `_tuned_retrained`: Re-training of a tuned model

---

**Quick Start**: Just run your experiments as normal. History files will be automatically created in the `training_history/` subdirectory of your results.

