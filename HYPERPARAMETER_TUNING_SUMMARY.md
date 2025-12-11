# Hyperparameter Tuning Process: Technical Summary

## Overview

The codebase implements a two-stage hyperparameter optimization framework using Optuna, a Bayesian optimization library. The tuning process is integrated into the unified experiment runner and applies to all neural network models in the study.

## Two-Stage Optimization Framework

### Stage 1: Architecture Parameter Search

The first stage optimizes model architecture hyperparameters while keeping training parameters at default values. Architecture parameters define the structural properties of the neural network.

**Implementation**: Function `run_optuna_stage()` in `evaluation/two_stage_hp_opt.py` (lines 145-236)

**Default Configuration**:
- Number of trials: 10 (configurable via `arch_trials` parameter)
- Cross-validation: StratifiedShuffleSplit with 1 split (80% train, 20% validation)
- Optimization direction: Maximize ROC-AUC score
- Random seed: User-specified for reproducibility

**Architecture Parameters by Model**:

1. **EEGNet** (lines 679-693):
   - `F1`: Number of temporal filters [4, 8, 16]
   - `D`: Depth multiplier [1, 2, 4]
   - `kernel_length`: Temporal convolution kernel size [64-256, step 32]
   - `drop_prob`: Dropout probability [0.1-0.5]

2. **REEGNet** (lines 663-676):
   - `lstm_hidden_size`: LSTM hidden units [8, 16, 32, 64]
   - `drop_prob`: Dropout probability [0.1-0.5]

3. **CNN-NCP** (lines 621-638):
   - `F1`: Number of temporal filters [4, 8, 16]
   - `D`: Depth multiplier [1, 2, 4]
   - `kernel_length`: Temporal convolution kernel size [64-256, step 32]
   - `ncp_hidden_dim`: NCP hidden layer size [19-128]
   - `sparsity`: Neural circuit sparsity [0.2-0.9]

4. **CNNCfC-v2 / Improved CNNCfC** (lines 545-597):
   - `F1`: Number of temporal filters [4, 8, 12, 16]
   - `D`: Depth multiplier [1, 2, 4]
   - `kernel_length`: Temporal convolution kernel size [64-256, step 16]
   - `temporal_kernel_size`: Temporal pooling kernel [3, 5, 7]
   - `temporal_stride`: Temporal pooling stride [2, 4, 6, 8]
   - `max_seq_length`: Maximum sequence length [150-1000]
   - `ncp_hidden_dim`: CfC hidden layer size [8-64]
   - `drop_prob`: Dropout probability [0.1-0.5]
   - `mixed_memory`: Use mixed memory mode [True, False]
   - `mode`: CfC cell mode ["default", "pure", "no_gate"]
   - `activation`: Activation function ["lecun_tanh", "silu", "relu", "tanh", "gelu"]
   - `backbone_units`: Backbone network units [16-256]
   - `backbone_layers`: Number of backbone layers [1-3]
   - `backbone_dropout`: Backbone dropout [0.0-0.5]

5. **CNN Small World** (lines 374-437):
   - CNN parameters (F1, D, kernel_length) as above
   - `temporal_kernel_size`: [3, 5, 7, 9]
   - `temporal_stride`: [2, 4, 6, 8]
   - `max_seq_length`: [150-500]
   - `ncp_hidden_dim`: [16-128]
   - `drop_prob`: [0.1-0.5]
   - `n_modules`: Number of small-world modules [1, 2, 4, 6, 8]
   - `rewiring_prob`: Small-world rewiring probability [0.1-0.5]

6. **SPP-NCP** (lines 641-660):
   - `ncp_hidden_dim`: [11-16] (narrower range for efficiency)
   - `sparsity`: [0.4-0.9]
   - `drop_prob`: [0.1-0.5]

### Stage 2: Training Parameter Search

The second stage optimizes training hyperparameters while holding architecture parameters fixed at their Stage 1 optimal values.

**Implementation**: Function `run_optuna_stage()` with modified parameter space (lines 793-820)

**Default Configuration**:
- Number of trials: 10 (configurable via `train_trials` parameter)
- Same cross-validation strategy as Stage 1
- Architecture parameters frozen from Stage 1 results

**Training Parameters (Common across models)**:
- `optimizer__lr`: Learning rate, log-uniform [1e-6, 1e-2]
- `optimizer__weight_decay`: L2 regularization, log-uniform [1e-6, 1e-2]
- `batch_size`: Batch size [4, 8, 16, 32, 64]

**Parameter Space Type**: Optuna's `suggest_loguniform` is used for learning rate and weight decay to sample more densely in lower ranges where optimal values are typically found.


## Cross-Validation Strategy

### Hyperparameter Optimization CV

During hyperparameter tuning (both Stage 1 and Stage 2), the system uses:
- **Splitter**: `StratifiedShuffleSplit` from scikit-learn
- **Configuration**: 1 split, 20% test size
- **Purpose**: Provides a single train/validation split for evaluating hyperparameter configurations
- **Implementation**: Line 199 in `evaluation/two_stage_hp_opt.py`

### Training Loop

**Function**: `unified_cv_training_loop_method()` (lines 110-142)

**Process**:
1. For each CV fold:
   - Set model to training mode
   - Fit model on training partition
   - Set model to evaluation mode
   - Compute predictions on validation partition
   - Calculate ROC-AUC score
2. Return mean ROC-AUC across all folds

**Model State Management**:
- Explicit `model.module_.train()` and `model.module_.eval()` calls
- `torch.no_grad()` context during evaluation
- Model re-initialized for each Optuna trial

## Evaluation Metrics

**Primary Metric**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

**Handling of Multi-class Problems**:
- Binary classification (e.g., BNCI2014_001 motor imagery): Uses positive class probabilities
- Multi-class classification (e.g., Lee2019_SSVEP): Uses One-vs-Rest (OvR) strategy
- Implementation: Lines 84-91 in `two_stage_hp_opt.py`

**Rationale**: ROC-AUC is insensitive to class imbalance and provides a threshold-independent measure of classification performance.

## Integration with Main Experiment

### Invocation

The hyperparameter tuning process is invoked when the `--tune` flag is set in the unified experiment runner.

**Entry Point**: `UnifiedExperimentRunner._run_hyperparameter_optimization()` (lines 530-626 in `unified_experiment_runner.py`)

**Process**:
1. Extract training data from current CV fold
2. Call two-stage optimization with appropriate model function and parameter spaces
3. Apply best parameters to fresh model instance
4. Train final model on full training set
5. Evaluate on held-out validation set
6. Save training history

### Data Flow

**Within-Session Evaluation**:
- Each session (0train, 1test) processed separately
- For each session: 5-fold cross-validation
- Each fold: hyperparameter optimization on fold's training data
- Final evaluation on fold's validation data

**Cross-Session Evaluation**:
- Leave-one-group-out by session
- Each left-out session: hyperparameter optimization on remaining sessions' data
- Evaluation on held-out session

**Cross-Subject Evaluation**:
- Custom 3-fold subject split (approximately 1/3 subjects per fold)
- Each fold: hyperparameter optimization on 2/3 of subjects
- Evaluation on remaining 1/3 of subjects

## Noise-Aware Hyperparameter Optimization

The system supports hyperparameter optimization for noise-robust training strategies.

**Implementation**: Function `alternate_two_stage_optuna()` (lines 697-760)

**Modes**:
1. **Perturb Mode** (`mode="perturb"`):
   - Training data augmented with noise during model training
   - Uses `TrainOnlyNoiseClassifier` wrapper
   - Optimization evaluates performance under noisy training conditions

2. **Augment Mode** (`mode="augment"`):
   - Training data concatenated with noisy versions
   - Uses `ConcatenatedNoiseAugmenter` wrapper
   - Uses `GroupKFold` (n_splits=3) to prevent data leakage between clean and augmented samples
   - Implementation: Lines 285-288 in `two_stage_hp_opt.py`

**Noise Parameters**:
- `noise_type`: Type of noise (gaussian, dropout, eog)
- `intensity`: Noise strength parameter

**Model Wrapping**: Lines 363-396 in `unified_experiment_runner.py` show how base models are wrapped with noise augmentation layers.

## Training Configuration

### Maximum Epochs
- Default: 200 epochs (`DEFAULT_MAX_EPOCHS` in `globals.py`, line 53)
- Increased from 100 to allow sufficient training time
- Safety check: Prevents configurations > 200 epochs (lines 192-195 in `two_stage_hp_opt.py`)

### Early Stopping
- **Monitor**: Validation loss (`valid_loss`)
- **Patience**: 20 epochs
- **Threshold**: 1e-5 (minimum improvement to reset patience counter)
- **Load Best**: True (restore best model weights)
- Implementation: `get_early_stopping_callback()` in `globals.py` (lines 38-50)

### Underfitting Detection
- **Threshold**: 0.70 ROC-AUC
- **Action**: If clean validation score < 0.70, retrain without early stopping
- **Implementation**: Lines 792-824 in `unified_experiment_runner.py`
- **Rationale**: Early stopping may terminate training prematurely for difficult subjects/sessions

### Model Initialization
- Fresh model initialization for each Optuna trial
- Deterministic initialization using fixed random seeds
- Seeds control: PyTorch, NumPy, CUDA operations
- Implementation: `set_seeds()` in `globals.py` (lines 9-25)

## Parameter Storage and Retrieval

### Parameter Formatting
Function `format_params()` (lines 22-38 in `two_stage_hp_opt.py`) handles parameter name prefixing:
- Module parameters: Prefixed with `module__` (e.g., `module__F1`)
- Optimizer parameters: Prefixed with `optimizer__` (e.g., `optimizer__lr`)
- Other parameters: Prefixed with base prefix (e.g., `base_pipeline__` for wrapped models)

### Best Parameters Extraction
After optimization completes:
1. Extract best parameters from Optuna study
2. Apply proper prefixing based on model type
3. Merge architecture and training parameters
4. Store in results dataframe for reproducibility

Implementation: Lines 509-528 in `unified_experiment_runner.py`

## Adaptive Parameter Spaces

The codebase includes experimental adaptive parameter spaces that adjust search ranges based on previous trial results.

**Implementation**: Functions `adaptive_improved_cnncfc_architecture_space()` and related (lines 1041-1256 in `two_stage_hp_opt.py`)

**Features**:
- After 10 trials, narrows search space around promising regions
- Adjusts ranges for `ncp_hidden_dim`, `F1`, and other key parameters
- Currently not used in main experiment pipeline

## Computational Considerations

### Trial Budgets
- Architecture trials: 10-20 (default: 10)
- Training trials: 10-20 (default: 10)
- Joint trials: 10 (when enabled)
- Total trials per tuning session: 20-50

### Parallelization
- Optuna studies are sequential within a single experiment
- Parallelization achieved at the experiment level (multiple subjects/seeds run concurrently)
- Each experiment maintains its own Optuna study

### Resource Management
- HDF5 checkpoint files created with unique IDs
- Temporary directories cleaned up after subject completion
- Model checkpoints not saved during hyperparameter optimization (only final models)

## Output and Logging

### Optuna Study Artifacts
For each tuning session, the following files are saved:
- `optuna_study.pkl`: Serialized Optuna study object
- Location: `{output_dir}/Optuna/fold_{fold_idx}/architecture/` and `.../training/`

### Training History
After tuning completes:
- Full epoch-by-epoch training history saved as JSON
- Location: `{output_dir}/training_history/`
- Filename format: `history_sub{subject}_sess{session}_fold{fold}_mode_tuned.json`
- Implementation: `save_training_history()` function (lines 129-198 in `unified_experiment_runner.py`)

### Results Files
Final results include:
- Subject ID, session, fold index
- Clean and corrupted validation scores (for test_perturb mode)
- All hyperparameters used (architecture + training)
- Training time, evaluation time
- Model configuration parameters

## Design Decisions and Rationale

### Two-Stage vs. Joint Optimization
**Decision**: Use two-stage optimization by default

**Rationale**:
1. Architecture parameters typically have larger impact on performance than training parameters
2. Separating stages allows more trials for architecture search
3. Reduced search space in Stage 2 enables faster convergence
4. Total trials (20-40) comparable to joint optimization but better coverage

### Single Split for HP Optimization
**Decision**: Use 1 split (80/20) rather than k-fold during hyperparameter search

**Rationale**:
1. Computational efficiency: Each Optuna trial trains 1 model instead of k models
2. Sufficient for ranking hyperparameter configurations
3. Final evaluation uses proper cross-validation (5-fold or leave-one-out)
4. Allows 2-3x more Optuna trials within same time budget

### Log-Uniform Sampling for Learning Rate
**Decision**: Use `suggest_loguniform` for learning rate and weight decay

**Rationale**:
1. Optimal learning rates typically in range [1e-5, 1e-3]
2. Linear sampling would under-sample this critical region
3. Log-uniform provides equal density in logarithmic space
4. Standard practice in deep learning hyperparameter optimization

### ROC-AUC as Primary Metric
**Decision**: Optimize ROC-AUC rather than accuracy

**Rationale**:
1. Robust to class imbalance
2. Threshold-independent measure
3. Directly relevant to medical/clinical applications
4. Enables comparison across datasets with different class distributions

### Early Stopping Configuration
**Decision**: Patience=20, threshold=1e-5, monitor=valid_loss

**Rationale**:
1. Higher patience prevents premature stopping
2. Small threshold allows continued training unless truly plateaued
3. Valid_loss more stable than valid_acc for early stopping signal
4. load_best ensures optimal model weights retained

### Underfitting Mitigation
**Decision**: Re-train without early stopping if ROC-AUC < 0.70

**Rationale**:
1. Some subjects/sessions require more epochs to converge
2. Early stopping may be too aggressive for difficult cases
3. 0.70 threshold indicates model has not learned meaningful patterns
4. Re-training cost justified by improved performance

### Noise-Aware Tuning
**Decision**: Separate parameter spaces for perturb/augment modes

**Rationale**:
1. Noise-robust training may require different hyperparameters
2. Dropout probability may interact with injected noise
3. Learning rate may need adjustment for augmented data
4. Model capacity (hidden_dim) may differ for noisy vs. clean training

## Limitations and Assumptions

1. **Single Subject Optimization**: Hyperparameters optimized separately for each subject. No transfer of optimal parameters across subjects.

2. **Computational Cost**: Full hyperparameter tuning increases experiment time by 10-20x compared to fixed hyperparameters.

3. **Local Optima**: Bayesian optimization may converge to local optima. Multiple seeds partially mitigate this.

4. **Parameter Space Coverage**: Fixed parameter ranges may not be optimal for all datasets/subjects. Adaptive spaces partially address this but are not currently used.

5. **Train/Test Leakage**: During Stage 1 and Stage 2, the same validation set is used multiple times. This is acceptable because final evaluation uses separate held-out data.

6. **Stochasticity**: Despite deterministic settings, some GPU operations may introduce minor variability. This is mitigated by using multiple seeds.

## Summary

The hyperparameter tuning system implements a rigorous two-stage Bayesian optimization approach using Optuna. Architecture parameters are optimized first, followed by training parameters with frozen architecture. The system uses stratified splitting, ROC-AUC as the optimization metric, and integrates seamlessly with the main experiment pipeline. Special provisions handle noise-aware training modes, underfitting detection, and comprehensive logging of all hyperparameters and training history. The design balances computational efficiency with thorough exploration of the hyperparameter space.

