# Experimental Evaluation Pipeline: Technical Summary (test_perturb Mode)

## Overview

The experimental evaluation pipeline is implemented in `evaluation/unified_experiment_runner.py` and provides a unified framework for evaluating EEG classification models under various conditions. This document focuses specifically on the `test_perturb` mode, which evaluates model robustness by testing performance on clean data and then systematically perturbing validation data with increasing noise intensities.

The pipeline supports multiple evaluation strategies (WithinSession, CrossSession, CrossSubject), multiple neural network architectures, and integrates with the hyperparameter tuning system described in the companion document.

## Core Architecture

### UnifiedExperimentRunner Class

**Implementation**: Lines 201-1141 in `unified_experiment_runner.py`

The `UnifiedExperimentRunner` class encapsulates all experiment logic and provides a consistent interface across different evaluation modes, datasets, and experimental configurations.

**Key Components**:
- Dataset and paradigm initialization
- Model instantiation with dynamic dimensions
- Cross-validation strategy selection
- Training and evaluation loops
- Result aggregation and storage
- Training history logging

### Entry Point

**Function**: `main()` (lines 1143-1269)

**Command-line Interface**:
```bash
python evaluation/unified_experiment_runner.py \
  --model {model_name} \
  --dataset {dataset_name} \
  --subjects {subject_ids} \
  --mode test_perturb \
  --eval_mode {WithinSession|CrossSession|CrossSubject} \
  --seed {seed} \
  [--tune] \
  [--overwrite]
```

**Parameters**:
- `--model`: Model architecture (eegnet, reegnet, cnn_ncp, etc.)
- `--dataset`: Dataset name (BNCI2014_001 or Lee2019_SSVEP)
- `--subjects`: Space-separated list of subject IDs
- `--mode`: Experiment mode (test_perturb for robustness evaluation)
- `--eval_mode`: Cross-validation strategy
- `--seed`: Random seed for reproducibility
- `--tune`: Enable hyperparameter optimization (optional)
- `--overwrite`: Overwrite existing results (optional)

## Data Flow

### 1. Initialization Phase

**Method**: `__init__()` (lines 206-258)

**Process**:
1. Validate experiment parameters
2. Set random seeds for reproducibility (`set_seeds()` from `globals.py`)
3. Initialize dataset object (BNCI2014_001 or Lee2019_SSVEP)
4. Create paradigm object (MotorImagery or SSVEP)
5. Filter dataset to specified subjects
6. Create output directory structure

**Output Paths**:
- Results: `results/{paradigm}/{dataset}/{model}/{eval_mode}/{seed}/sub-{subject}/{session}/{mode}/`
- Training history: `{results_path}/training_history/`
- Hyperparameter optimization: `{results_path}/Optuna/fold_{fold_idx}/`

### 2. Data Loading and Preprocessing

**Method**: `_determine_data_dimensions()` (lines 325-337)

**Process**:
1. Load sample data from first subject
2. Extract channel count (`n_chans`) and time points (`n_times`)
3. These dimensions are used to instantiate models with correct input shapes

**Dataset Characteristics**:
- **BNCI2014_001**: 22 channels, 1000 time points (4 seconds @ 250 Hz), 2 classes
- **Lee2019_SSVEP**: 62 channels, 1000 time points (4 seconds @ 250 Hz), 4 classes

### 3. Cross-Validation Setup

**Method**: `prepare_data_cv()` (lines 399-433)

The system uses different cross-validation strategies based on the evaluation mode:

#### WithinSession Evaluation
- **Splitter**: `StratifiedKFold` with 5 folds
- **Configuration**: Shuffle=True, random_state=seed
- **Purpose**: Evaluate generalization within a single session
- **Process**: Each session (0train, 1test) processed independently
- **Implementation**: Lines 408-414

#### CrossSession Evaluation
- **Splitter**: `LeaveOneGroupOut` by session
- **Configuration**: Groups defined by session labels
- **Purpose**: Evaluate generalization across different recording sessions
- **Process**: Each session held out once as test set
- **Implementation**: Lines 416-421

#### CrossSubject Evaluation
- **Splitter**: `ThreeFoldSubjectSplit` (custom)
- **Configuration**: 3 folds, approximately 1/3 subjects per fold
- **Purpose**: Evaluate generalization across different subjects
- **Process**: All subjects' data pooled, then split by subject identity
- **Implementation**: Lines 66-126 (class definition), 423-429 (usage)

**Custom ThreeFoldSubjectSplit**:
- Divides subjects into 3 groups of equal size
- Each fold uses 2/3 subjects for training, 1/3 for testing
- Remainder subjects (if not divisible by 3) added to training in all folds
- Ensures no subject appears in both train and test within a fold

## test_perturb Mode: Detailed Implementation

### Overview

The `test_perturb` mode evaluates model robustness by:
1. Training models on clean data
2. Evaluating on clean validation data (baseline performance)
3. Systematically perturbing validation data with three noise types at multiple intensities
4. Recording performance degradation curves

### Training Phase

**Method**: `_train_and_evaluate_perturb()` (lines 745-831)

**Process**:
1. Create model instance with dimensions determined from data
2. Train model on clean training data (X_train, y_train)
3. Save training history to JSON file
4. Evaluate on clean validation data to establish baseline
5. Check for underfitting (ROC-AUC < 0.70 threshold)
6. If underfitting detected and `--tune` flag not set:
   - Remove EarlyStopping callback
   - Retrain model to completion
   - Re-evaluate on clean data
   - Use maximum of original and retrained scores

**Underfitting Detection** (lines 792-824):
- **Threshold**: 0.70 ROC-AUC
- **Rationale**: Scores below 0.70 indicate model has not learned meaningful patterns
- **Action**: Retrain without early stopping to allow full convergence
- **Note**: This re-training is skipped when `--tune` is enabled to reduce computational cost

### Perturbation Testing Phase

**Method**: `_evaluate_perturb()` (lines 686-743)

**Process**:
1. For each noise type (gaussian, dropout, eog):
   - Determine intensity bounds based on saturation analysis
   - Generate 20 intensity steps from minimum to saturation point
   - For each intensity:
     - Create perturbed validation data
     - Evaluate model on perturbed data
     - Record metrics (ROC-AUC, accuracy, precision, recall, F1)
     - Calculate performance drop relative to clean baseline

**Noise Types**:

1. **Gaussian Noise** (lines 736-778 in `augmentation/noise.py`):
   - Adds magnitude-aware Gaussian noise to EEG channels
   - Intensity parameter: Percentage of noise relative to signal RMS
   - Implementation: `_improved_apply_gaussian_noise()`
   - Noise scale: 4.0 × signal_rms × (intensity / 100)
   - Progressive channel contamination based on intensity

2. **Dropout Noise** (lines 714-724 in `augmentation/noise.py`):
   - Randomly zeros out EEG channels
   - Intensity parameter: Percentage of channels to drop (0-100)
   - Implementation: `_apply_channel_dropout()`
   - Drops at least 1 channel if intensity > 0
   - Different channels dropped for each epoch

3. **EOG (Electrooculogram) Artifacts** (lines 838-883 in `augmentation/noise.py`):
   - Injects realistic eye movement artifacts
   - Intensity parameter: Temporal coverage (percentage of time with artifacts)
   - Implementation: `_apply_realistic_eog_noise()`
   - Uses learned generic EOG mixing template
   - Includes blinks, saccades, and drift components

### Dynamic Intensity Bounds

**Function**: `get_noise_intensities()` (lines 105-119 in `utils.py`)

**Saturation-Based Bounds**:
- Reads saturation points from `saturation_results/saturation_points_summary.csv`
- Sets minimum intensity to 1.0 (subtle perturbation)
- Sets maximum intensity to saturation point (where model performance plateaus)
- Generates 20 evenly-spaced intensity steps

**Default Bounds** (if saturation data unavailable):
- Minimum: 1.0
- Maximum: 50.0

**Example Saturation Points**:
- BNCI2014_001 + dropout: 50.0
- Lee2019_SSVEP + dropout: 100.0
- BNCI2014_001 + gaussian: 100.0
- Lee2019_SSVEP + gaussian: 100.0
- BNCI2014_001 + eog: 100.0
- Lee2019_SSVEP + eog: 100.0

### Realistic EOG Artifact Generation

**Implementation**: Lines 353-630 in `augmentation/noise.py`

**EOG Template System**:
1. **Template Source**: Pre-computed from real EEG-EOG dataset
2. **Template Path**: `notebooks/eog_mixing_results/generic_eog_mixing_template.npz`
3. **Template Components**:
   - Mixing matrix (19 channels × 2 regressors): Maps VEOG/HEOG to EEG channels
   - VEOG standard deviation: Calibration for vertical eye movements
   - HEOG standard deviation: Calibration for horizontal eye movements
   - Target RMS median: Overall artifact strength calibration

**Artifact Generation Process**:
1. **Blink Template Creation** (lines 481-492):
   - Duration: 100-300 ms (variable)
   - Shape: Gaussian-like with peak at 1/3 to 2/3 through duration
   - Normalized to unit amplitude

2. **Temporal Coverage Control** (lines 494-516):
   - Calculate number of blinks to achieve target temporal coverage
   - Example: 10% coverage with 200ms blinks ≈ 5 blinks per 10-second window
   - Ensure coverage does not exceed physically realistic limits

3. **Blink Placement** (lines 517-548):
   - Boundary intersection allowed (default): Blinks can start before or end after sample boundaries
   - Creates partial blinks for increased variability
   - Mix of evenly-distributed and random placement

4. **Realistic Components Added** (lines 555-624):
   - **Slow drift**: 1-3 low-frequency components (0.3-2.5 Hz) simulating eye position drift
   - **Amplitude variability**: Each blink has 60-160% of baseline amplitude
   - **Lateral component**: HEOG has 25% of VEOG amplitude with random direction
   - **Microsaccades**: 40% of blinks include brief eye movements (50ms, 8-15% amplitude)
   - **Blink clusters**: 25% chance of 1-2 additional blinks in 250-500ms succession

5. **Spatial Projection** (lines 265-272):
   - Interpolates mixing matrix to target channel montage if needed
   - Applies mixing matrix: `eog_artifacts = mixing_matrix @ [veog_tc, heog_tc]`
   - Scales artifacts by calibration factor (15000.0 default)

6. **Artifact Injection** (lines 288-289):
   - Adds projected artifacts to clean EEG data
   - Preserves original units (microvolts or volts)

**Design Rationale**:
- Temporal coverage parameter allows systematic robustness testing
- Multiple realistic components (drift, microsaccades, clusters) increase challenge
- Boundary intersection creates natural variability across epochs
- Template-based approach ensures physiologically realistic artifacts

## Integration with Hyperparameter Tuning

### Tuning in test_perturb Mode

**Method**: `_run_hyperparameter_optimization()` (lines 530-626)

**When Enabled** (`--tune` flag):
1. Hyperparameter optimization runs on fold's training data
2. Best parameters identified via two-stage Optuna optimization
3. Final model trained with best parameters on full training set
4. Clean validation score computed
5. Perturbation testing proceeds with tuned model

**Data Isolation**:
- Hyperparameter search: Uses internal 80/20 split of training data
- Final evaluation: Uses completely held-out validation fold
- No data leakage between optimization and evaluation

**Computational Cost**:
- With tuning: ~20-40 Optuna trials per fold
- Without tuning: Use default hyperparameters
- Tuning increases runtime by 10-20× but improves model quality

## Evaluation Metrics

### Primary Metrics

**Implementation**: `compute_classification_metrics()` in `evaluation/metrics.py`

**Metrics Computed**:
1. **ROC-AUC**: Area under ROC curve (primary metric)
   - Binary classification: Uses positive class probabilities
   - Multi-class: Uses One-vs-Rest (OvR) strategy
   - Range: [0.0, 1.0], where 0.5 = random, 1.0 = perfect

2. **Accuracy**: Proportion of correct predictions
   - Threshold: 0.5 for binary, argmax for multi-class

3. **Precision**: True positives / (True positives + False positives)
   - Macro-averaged for multi-class

4. **Recall**: True positives / (True positives + False negatives)
   - Macro-averaged for multi-class

5. **F1 Score**: Harmonic mean of precision and recall
   - Macro-averaged for multi-class

### Performance Drop Calculation

**Implementation**: Line 718 in `unified_experiment_runner.py`

**Formula**:
```
relative_drop = (clean_score - corrupted_score) / clean_score
```

**Interpretation**:
- 0.0 = No performance drop
- 0.2 = 20% performance drop
- 1.0 = Complete performance collapse

**Use Cases**:
- Comparing robustness across models
- Identifying saturation points
- Quantifying noise sensitivity

## Result Aggregation and Storage

### Per-Fold Results

**Data Structure** (lines 720-741):

For each perturbation (noise_type × intensity × fold):
```python
{
    'fold_idx': int,
    'noise_type': str,
    'intensity': float,
    'clean_score': float,  # Baseline ROC-AUC on clean data
    'corrupted_score': float,  # ROC-AUC on perturbed data
    'clean_roc_auc': float,
    'clean_accuracy': float,
    'clean_precision': float,
    'clean_recall': float,
    'clean_f1': float,
    'corrupted_roc_auc': float,
    'corrupted_accuracy': float,
    'corrupted_precision': float,
    'corrupted_recall': float,
    'corrupted_f1': float,
    'relative_drop': float,
    'training_time': float,  # Seconds
    'evaluation_time': float,  # Seconds
    'total_time': float,  # Seconds
    'session': str,
    'subject': int
}
```

### Fold Aggregation

**Method**: `_aggregate_fold_results()` (lines 1023-1104)

**WithinSession Mode** (lines 1030-1092):
- Groups results by (session, intensity, noise_type)
- Computes mean corrupted_score across 5 folds
- Computes mean clean_score across 5 folds
- Computes mean relative_drop across 5 folds
- One aggregated row per (subject, session, noise_type, intensity)

**CrossSession Mode** (lines 1094-1097):
- Each fold corresponds to a different session
- No aggregation needed (one evaluation per session)
- Simply drops fold_idx column

**CrossSubject Mode** (lines 1099-1102):
- Each fold represents different group of evaluation subjects
- Keeps folds separate (each fold = different subject group)
- Adds eval_subjects metadata (comma-separated subject IDs)

### File Output

**Method**: `_save_results()` (lines 1106-1140)

**File Naming Convention**:
```
results/{paradigm}/{dataset}/{model}/{eval_mode}/{seed}/
  sub-{subject:03d}/{session}/{mode}/
    {model}_{mode}_{noise_type}_{intensity}_subject_{subject:03d}_seed{seed}.csv
```

**Example**:
```
results/MotorImagery/BNCI2014_001/eegnet/CrossSessionEvaluation/42/
  sub-001/0train/test_perturb/
    eegnet_test_perturb_gaussian_10.0_subject_001_seed42.csv
```

**CSV Structure**:
- Each row: One result entry
- Columns: All metrics + metadata (subject, session, model, mode, seed, hyperparameters)
- For test_perturb: Multiple rows per subject/session (one per noise_type × intensity)

### Training History Storage

**Function**: `save_training_history()` (lines 129-198)

**Saved Information**:
- Epoch number
- Train loss, validation loss (if applicable)
- Train accuracy, validation accuracy
- Learning rate (if scheduled)
- Duration per epoch
- Any custom metrics tracked during training

**File Location**:
```
results/{paradigm}/{dataset}/{model}/{eval_mode}/{seed}/
  sub-{subject:03d}/{session}/{mode}/training_history/
    history_sub{subject:03d}_sess{session}_fold{fold}_{mode}.json
```

**Format**: JSON array of epoch dictionaries

**Example**:
```json
[
  {
    "epoch": 1,
    "train_loss": 0.693,
    "valid_loss": 0.685,
    "train_acc": 0.512,
    "valid_acc": 0.523,
    "dur": 2.34
  },
  ...
]
```

## Experiment Execution Flow

### Single Subject, WithinSession, test_perturb

1. **Setup**:
   - Load subject data (X, y, metadata)
   - Encode labels (y_encoded)
   - Create 5-fold stratified splitter

2. **For each session** (0train, 1test):
   - Filter data to session
   - **For each fold** (1-5):
     - Split session data into train/validation
     - **If --tune enabled**:
       - Run hyperparameter optimization on train data
       - Train final model with best parameters
     - **Else**:
       - Train model with default parameters
     - Save training history
     - Evaluate on clean validation data (baseline)
     - **If underfitting detected** (score < 0.70 and not tuning):
       - Retrain without early stopping
       - Re-evaluate
     - **For each noise type** (gaussian, dropout, eog):
       - **For each intensity** (20 steps from min to saturation):
         - Perturb validation data
         - Evaluate on perturbed data
         - Record all metrics

3. **Aggregation**:
   - Group results by (session, noise_type, intensity)
   - Compute mean metrics across 5 folds
   - Save aggregated results to CSV

4. **Cleanup**:
   - Remove temporary HDF5 checkpoint files

### Multiple Subjects, CrossSubject, test_perturb

1. **Setup**:
   - Load all subjects' data at once (X, y, metadata)
   - Encode labels
   - Create 3-fold subject splitter

2. **For each fold** (1-3):
   - Split data by subject groups
   - Identify evaluation subjects for this fold
   - **If --tune enabled**:
     - Run hyperparameter optimization on training subjects
     - Train final model with best parameters
   - **Else**:
     - Train model with default parameters
   - Save training history
   - Evaluate on clean data from evaluation subjects (baseline)
   - **For each noise type** (gaussian, dropout, eog):
     - **For each intensity** (20 steps):
       - Perturb validation data
       - Evaluate
       - Record metrics

3. **Storage**:
   - Save results with eval_subjects metadata
   - Each fold stored separately (different subject groups)
   - No aggregation across folds

## Design Decisions and Rationale

### Separate Clean Score for Each Fold

**Decision**: Compute clean baseline for each fold independently

**Rationale**:
1. Different folds may have different validation subjects/sessions
2. Model trained on different data in each fold
3. Prevents cross-fold contamination
4. Allows fold-specific relative_drop calculations

### Underfitting Detection and Mitigation

**Decision**: Re-train without early stopping if ROC-AUC < 0.70 (when not tuning)

**Rationale**:
1. Some subjects/sessions are inherently difficult
2. Early stopping may be too conservative for difficult cases
3. 0.70 threshold indicates model has not learned meaningful patterns
4. Full training run ensures model reaches its potential
5. Skipped during tuning to reduce computational cost

### 20 Intensity Steps

**Decision**: Test 20 evenly-spaced intensities from minimum to saturation

**Rationale**:
1. Provides fine-grained degradation curves
2. Captures both subtle and severe perturbations
3. Enables accurate saturation point detection
4. Sufficient resolution for statistical analysis
5. Computationally feasible (20 × 3 = 60 perturbations per fold)

### Three Noise Types

**Decision**: Test gaussian, dropout, and eog noise

**Rationale**:
1. **Gaussian**: Represents sensor noise, electromagnetic interference
2. **Dropout**: Represents channel failures, bad electrode contact
3. **EOG**: Represents physiological artifacts (most common in real-world EEG)
4. These three cover distinct failure modes
5. EOG is most challenging due to non-stationary and structured nature

### Temporal Coverage for EOG

**Decision**: Use temporal coverage (not prevalence) for EOG intensity in test_perturb

**Rationale**:
1. Temporal coverage directly relates to real-world artifact burden
2. 10% coverage = 10% of recording time has artifacts (realistic)
3. Allows direct interpretation: "Model maintains X% performance with Y% EOG contamination"
4. Differs from prevalence (percentage of epochs) which is less interpretable

### Magnitude-Aware Gaussian Noise

**Decision**: Scale Gaussian noise relative to signal RMS (4.0 × signal_rms × intensity/100)

**Rationale**:
1. Ensures noise strength adapts to signal magnitude
2. Prevents too-weak or too-strong noise for different datasets
3. Makes intensity parameter interpretable across datasets
4. 4.0 multiplier chosen empirically to create challenging but not overwhelming noise

### Saturation-Based Bounds

**Decision**: Use dataset-specific and noise-specific intensity bounds

**Rationale**:
1. Different noise types have different severity profiles
2. Different datasets have different noise sensitivities
3. Testing beyond saturation point wastes computation
4. Pre-computed saturation points guide efficient intensity selection
5. Enables fair cross-model comparisons at equivalent difficulty levels

### Model State Management

**Decision**: Explicit `model.module_.train()` and `model.module_.eval()` calls with `torch.no_grad()` during evaluation

**Rationale**:
1. Ensures dropout and batch normalization behave correctly
2. Prevents gradient accumulation during evaluation (memory efficiency)
3. Matches PyTorch best practices
4. Critical for reproducibility with stochastic layers

### CrossSubject 3-Fold Split

**Decision**: Use 3-fold instead of standard 5-fold or 10-fold

**Rationale**:
1. Limits number of subjects: 9 subjects ÷ 3 = 3 subjects per fold
2. Provides reasonable train/test ratio (2/3 : 1/3)
3. Reduces computational cost compared to leave-one-subject-out
4. Still provides multiple validation points (3 folds)
5. Balances statistical power with feasibility

## Computational Considerations

### Memory Management

**Per-Epoch Memory**:
- BNCI2014_001: 22 channels × 1000 samples × 4 bytes = ~88 KB per epoch
- Lee2019_SSVEP: 62 channels × 1000 samples × 4 bytes = ~248 KB per epoch
- Typical dataset: 288 epochs × 88 KB = ~25 MB per subject

**GPU Memory**:
- Model parameters: 10-100 KB (small models)
- Batch processing: Batch_size × epoch_size × gradient storage
- Typical: < 500 MB GPU memory for batch_size=32

### Runtime Estimates

**Without Hyperparameter Tuning**:
- Single fold training: 2-5 minutes
- Single fold evaluation (clean): 5-10 seconds
- Single fold evaluation (60 perturbations): 1-2 minutes
- Total per subject (WithinSession): 2 sessions × 5 folds × 7 minutes ≈ 70 minutes

**With Hyperparameter Tuning**:
- Single fold optimization: 20-40 trials × 2-3 minutes = 40-120 minutes
- Single fold training with best params: 2-5 minutes
- Single fold evaluation: Same as above
- Total per subject (WithinSession): 2 sessions × 5 folds × 50 minutes ≈ 500 minutes

**CrossSubject**:
- More training data (all subjects pooled): 2-3× longer training time
- Fewer folds (3 vs 5): Reduces total time
- With tuning: 3 folds × 50 minutes ≈ 150 minutes per subject group

### Parallelization Strategy

**Current Implementation**:
- Sequential within subject (folds processed serially)
- Parallel across subjects (run multiple subjects concurrently)
- Parallel across seeds (run multiple seeds concurrently)

**Typical Batch Job**:
- 9 subjects × 5 seeds = 45 independent jobs
- Each job: One subject, one seed, one eval_mode
- Cluster allocation: 45 CPUs or 45 GPU nodes
- Wall time: ~1-2 hours per job (without tuning), 8-12 hours (with tuning)

## Limitations and Assumptions

1. **Fixed Noise Models**: The three noise types (gaussian, dropout, eog) do not cover all possible real-world artifacts (muscle artifacts, electrode shifts, etc.)

2. **Static Corruption**: Noise is applied at test time only. Real-world systems may experience time-varying artifacts.

3. **Independent Epochs**: Each epoch perturbed independently. Real artifacts often have temporal structure across epochs.

4. **Saturation Points**: Pre-computed saturation points may not generalize to all models or subjects.

5. **Underfitting Threshold**: The 0.70 ROC-AUC threshold is empirically chosen and may not be optimal for all datasets.

6. **Deterministic Evaluation**: Despite seed setting, some GPU operations may introduce minor variability.

7. **No Online Adaptation**: Models are evaluated in a static manner without opportunity to adapt to corrupted inputs.

8. **Single Test Phase**: Models are not evaluated on clean data after exposure to corrupted data (no forgetting assessment).

## Summary

The experimental evaluation pipeline implements a comprehensive framework for assessing EEG classification model robustness in the `test_perturb` mode. Models are trained on clean data and evaluated on systematically perturbed validation sets across three noise types (gaussian, dropout, eog) at 20 intensity levels determined by saturation analysis. The pipeline supports three evaluation modes (WithinSession, CrossSession, CrossSubject) with appropriate cross-validation strategies. Integration with hyperparameter optimization enables fair comparison of tuned models. Results are aggregated across folds and stored in structured CSV files with comprehensive metadata. The design balances statistical rigor, computational efficiency, and real-world relevance.

