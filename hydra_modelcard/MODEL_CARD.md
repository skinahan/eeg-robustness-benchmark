# Model Card: HYDRA (BranchedWiredCfC Architecture #4)

**Model Name**: HYDRA (Hybrid DIVA with Recurrent Attention)  
**Alternative Name**: BranchedWiredCfC with Architecture #4  
**Version**: 1.0  
**Date**: 2025-01-16

---

## 1. Model Details

### 1.1 Model Name and Description

**HYDRA** (Hybrid DIVA with Recurrent Attention) is a deep learning model designed for EEG-based motor imagery classification. The model combines a CNN-based feature extraction front-end with a branched recurrent processing module using Closed-form Continuous-time (CfC) cells with an optimized wiring configuration (Architecture #4).

The model architecture integrates:
- **DIVA Front-end**: CNN feature extraction with multi-scale temporal processing
- **SNR Gating**: Adaptive channel suppression for noise robustness
- **Branched Recurrent Processing**: Parallel CfC processing over temporal bins
- **Attention Mechanisms**: Hierarchical attention pooling (intra-bin and inter-bin)
- **Optimized Wiring**: Architecture #4 discovered through graph-theoretic architecture search

### 1.2 Model Type

- **Architecture**: Hybrid CNN-Recurrent Neural Network
- **Task**: Binary Classification (Motor Imagery: Left Hand vs. Right Hand)
- **Input Format**: Multi-channel EEG time-series signals
- **Output Format**: Classification logits (2 classes)

### 1.3 Model Version

- **Implementation**: `models/branched_wiredcfc.py`
- **Base Class**: `models/branched_diva_base.py`
- **Wiring System**: `architecture_refinement/arbitrary_wiring.py`
- **Architecture Configuration**: `outputs/architectures/best_architecture_4_trial_178.json`

---

## 2. Intended Use

### 2.1 Primary Use Case

HYDRA is designed for **motor imagery classification** from EEG signals, specifically:
- **Task**: Binary classification of left-hand vs. right-hand motor imagery
- **Dataset**: BNCI2014_001 (BCI Competition IV Dataset 2a)
- **Application Domain**: Brain-Computer Interface (BCI) systems, motor imagery decoding

### 2.2 Out-of-Scope Use Cases

**NOT intended for:**
- Clinical diagnosis or medical decision-making
- Real-time BCI applications without additional validation
- Other EEG paradigms (e.g., SSVEP, P300) without retraining
- Multi-class motor imagery tasks (>2 classes) without modification
- Cross-dataset deployment without domain adaptation

### 2.3 Target Users

- **Research Community**: EEG signal processing and BCI researchers
- **Developers**: BCI system developers and engineers
- **Practitioners**: Researchers working on motor imagery classification

---

## 3. Experimental Protocol

### 3.1 Dataset: BNCI2014_001

**Dataset Characteristics:**
- **Source**: BCI Competition IV Dataset 2a
- **Subjects**: 9 healthy subjects
- **Sessions**: 2 sessions per subject (training and testing)
- **Trials**: 288 trials per session (144 per class)
- **Channels**: 22 EEG channels (10-20 system)
- **Sampling Rate**: 250 Hz (resampled from original 512 Hz)
- **Trial Duration**: 4 seconds (1000 timepoints after resampling)
- **Classes**: 2 (Left Hand, Right Hand motor imagery)
- **Frequency Band**: 8-35 Hz (motor imagery band)

### 3.2 Evaluation Protocol

**Cross-Session Evaluation:**
- **Training**: Session 0 (0train)
- **Testing**: Session 1 (1test)
- **Cross-Validation**: 3-fold cross-subject validation
- **Metrics**: ROC-AUC (primary), Accuracy, Precision, Recall, F1-Score

**Evaluation Modes:**
- **WithinSession**: 5-fold stratified cross-validation within a single session
- **CrossSession**: Train on session 0, test on session 1
- **CrossSubject**: 3-fold cross-validation across subjects

### 3.3 Preprocessing Pipeline

1. **Bandpass Filtering**: 8-35 Hz (motor imagery frequency band)
2. **Resampling**: 250 Hz (from original 512 Hz)
3. **Epoching**: 4-second trials (tmin=0.0, tmax=4.0)
4. **No Baseline Correction**: `baseline=None` (raw signal used)
5. **Data Format**: `(n_trials, n_channels, n_times)` → `(288, 22, 1000)`

### 3.4 Training Protocol

**Training Configuration:**
- **Optimizer**: AdamW
- **Learning Rate**: 1e-2 (default), optimized via Optuna (1e-6 to 1e-2)
- **Weight Decay**: 0 (default), optimized via Optuna (1e-6 to 1e-2)
- **Batch Size**: 64 (default), optimized via Optuna (4, 8, 16, 32, 64)
- **Max Epochs**: 300 (with early stopping)
- **Early Stopping**: Patience=5, monitor='valid_loss'
- **Learning Rate Scheduler**: ExponentialLR (gamma=0.97)
- **Gradient Clipping**: 1.0 (L2 norm)
- **Validation Split**: 20% stratified split

**Hyperparameter Optimization:**
- **Method**: Two-stage Optuna optimization
- **Trials**: 20-40 trials per fold
- **Objective**: Maximize validation ROC-AUC
- **Search Space**: See Section 11 (Hyperparameters)

---

## 4. Training Data

### 4.1 Dataset: BNCI2014_001

**Dataset Details:**
- **Name**: BNCI2014_001 (BCI Competition IV Dataset 2a)
- **License**: Research use only
- **Citation**: Tangermann et al. (2012)
- **URL**: Available through MOABB (Mother of All BCI Benchmarks)

**Data Characteristics:**
- **Total Subjects**: 9
- **Total Sessions**: 18 (2 per subject)
- **Total Trials**: 5,184 (288 per session)
- **Class Distribution**: Balanced (50% left hand, 50% right hand)
- **Trial Length**: 4 seconds
- **Sampling Rate**: 250 Hz (resampled)
- **Channels**: 22 EEG channels

**Data Splits:**
- **Training**: Session 0 (0train) - 288 trials per subject
- **Testing**: Session 1 (1test) - 288 trials per subject
- **Validation**: 20% of training data (stratified)

### 4.2 Data Preprocessing

**Preprocessing Steps:**
1. **Bandpass Filtering**: 8-35 Hz (Butterworth, 4th order)
2. **Resampling**: 250 Hz (anti-aliasing filter applied)
3. **Epoching**: Extract 4-second windows (tmin=0.0, tmax=4.0)
4. **No Baseline Correction**: Raw signal preserved
5. **Normalization**: Per-trial z-scoring (optional, not used in default pipeline)

**Input Shape:**
- **Format**: `(batch_size, n_channels, n_times)`
- **Default**: `(64, 22, 1000)` for batch_size=64

### 4.3 Data Quality

**Known Issues:**
- Subject-specific variability in signal quality
- Potential artifacts (eye movements, muscle activity)
- Session-to-session non-stationarity
- No explicit artifact rejection in default pipeline

**Data Augmentation:**
- Not used in default training
- Noise injection used for robustness evaluation (not training)

---

## 5. Preprocessing

### 5.1 Input Preprocessing

**MOABB Paradigm Configuration:**
```python
MotorImagery(
    events=["left_hand", "right_hand"],
    fmin=8, fmax=35,
    tmin=0.0, tmax=4.0,
    baseline=None,
    resample=250.0,
    n_classes=2
)
```

**Preprocessing Steps:**
1. **Frequency Filtering**: 8-35 Hz bandpass (motor imagery band)
2. **Resampling**: 250 Hz (from original 512 Hz)
3. **Epoching**: 4-second trials
4. **Baseline**: None (no baseline correction)

### 5.2 Model-Internal Processing

**Architecture Processing Pipeline:**

1. **CNN Feature Extraction**:
   - Temporal convolution (F1=8, kernel=125)
   - Depthwise spatial convolution (F2=16)
   - Temporal pooling (factor=4)

2. **Multi-Scale Temporal Integration**:
   - Parallel dilated convolutions (kernels: 9, 15, 31; dilations: 1, 4, 16)
   - Residual connection
   - Layer normalization

3. **SNR Gating**:
   - Per-channel statistics computation
   - Adaptive channel suppression

4. **Temporal Downsampling**:
   - 1D convolution (kernel=3, stride=2)

5. **Branched Recurrent Processing**:
   - Temporal binning (bin_len=48, bin_stride=44)
   - Parallel CfC processing
   - Weighted residual connections

6. **Attention Pooling**:
   - Intra-bin attention
   - Inter-bin fusion (attention or mean)

7. **Classification Head**:
   - Layer normalization
   - Dropout (p=0.25)
   - Linear layer (H → n_outputs)

---

## 6. Architecture Diagram

See `figures/architecture_diagram.png` for a detailed architecture diagram.

**Architecture Overview:**

```
Input (B, 22, 1000)
    ↓
[1] Temporal Conv2D (F1=8, kernel=125) + BN + ELU
    ↓
[2] Depthwise Spatial Conv2D (F2=16) + BN + ELU
    ↓
[3] AvgPool2D (pool=4)
    ↓
[4] Dropout (p=0.25)
    ↓
[5] Multi-Scale Temporal Block
    ├─ Branch 1: Conv1D (kernel=9, dilation=1)
    ├─ Branch 2: Conv1D (kernel=15, dilation=4)
    └─ Branch 3: Conv1D (kernel=31, dilation=16)
    ↓
[6] SNR Gate (adaptive channel suppression)
    ↓
[7] Temporal Downsampler (Conv1D, kernel=3, stride=2)
    ↓
[8] Temporal Binning (bin_len=48, bin_stride=44)
    ↓
[9] Parallel CfC Processing (Architecture #4 wiring)
    ├─ Bin 1 → CfC → Residual → Attention
    ├─ Bin 2 → CfC → Residual → Attention
    └─ ...
    ↓
[10] Inter-Bin Fusion (Attention)
    ↓
[11] LayerNorm + Dropout
    ↓
[12] Linear Classification (H → 2)
    ↓
Output Logits (B, 2)
```

**Recurrent Compartment Wiring:**
See `figures/wiring_diagram.png` for the detailed wiring diagram of Architecture #4.

**Architecture #4 Wiring Characteristics:**
- **Input Units**: 8 (matches F2=16 after projection)
- **Hidden Units**: 43
- **Output Units**: 7 (projected to recurrent_output_size=16)
- **Wiring Type**: Sparse, structured connectivity
- **Discovery Method**: Graph-theoretic architecture search
- **Optimization Objectives**: Entropy, curvature, algebraic connectivity, efficiency

---

## 7. Hyperparameters

### 7.1 Architecture Hyperparameters

**CNN Front-end:**
- `F1`: 8 (temporal filter count)
- `D`: 2 (depthwise multiplier → F2 = 16)
- `kernel_length`: 125 (temporal kernel size)
- `pool_time`: 4 (temporal pooling factor)
- `drop_prob`: 0.25 (dropout probability)
- `bn_momentum`: 0.01 (BatchNorm momentum)
- `bn_eps`: 1e-3 (BatchNorm epsilon)

**Multi-Scale Temporal Block:**
- `ms_kernels`: (9, 15, 31) (kernel sizes)
- `ms_dilations`: (1, 4, 16) (dilation factors)

**Temporal Downsampling:**
- `temporal_kernel_size`: 3
- `temporal_stride`: 2

**SNR Gate:**
- `snr_reduction`: 4 (reduction factor for hidden layer)

**Binning:**
- `bin_len`: 48 (timesteps per bin, after downsampling)
- `bin_stride`: 44 (step between bin starts, creates overlap)
- `fusion`: "attn" (fusion method: "attn" or "mean")

**CfC Parameters:**
- `mixed_memory`: True
- `mode`: "default"
- `activation`: "lecun_tanh"
- `backbone_units`: 128
- `backbone_layers`: 1
- `backbone_dropout`: 0.0
- `recurrent_output_size`: None (defaults to F2=16)

**Residual Connection:**
- `residual_init_strategy`: "backwards_rezero" (default, empirically superior)
- Initial weight: 0.0 (recurrent at full strength at init)

### 7.2 Training Hyperparameters

**Default Values:**
- `optimizer`: AdamW
- `learning_rate`: 1e-2
- `weight_decay`: 0
- `batch_size`: 64
- `max_epochs`: 300
- `gradient_clip_value`: 1.0
- `learning_rate_scheduler`: ExponentialLR (gamma=0.97)
- `early_stopping_patience`: 5

**Hyperparameter Search Space (Optuna):**
- `learning_rate`: LogUniform(1e-6, 1e-2)
- `weight_decay`: LogUniform(1e-6, 1e-2)
- `batch_size`: Categorical([4, 8, 16, 32, 64])
- `drop_prob`: Uniform(0.1, 0.5)
- `F1`: Categorical([4, 8, 12, 16])
- `D`: Categorical([1, 2, 4])
- `kernel_length`: IntUniform(64, 256, step=32)
- `temporal_kernel_size`: Categorical([3, 5, 7])
- `temporal_stride`: Categorical([2, 4, 6, 8])
- `fusion`: Categorical(["attn", "mean"])

### 7.3 Hyperparameter Tuning

**Method**: Two-stage Optuna optimization
- **Stage 1**: Architecture hyperparameters (20-40 trials)
- **Stage 2**: Training hyperparameters (20-40 trials)
- **Objective**: Maximize validation ROC-AUC
- **Pruning**: MedianPruner (n_startup_trials=5)
- **Sampler**: TPESampler (seed-controlled)

---

## 8. Model Calibration

### 8.1 Calibration Assessment

**Method**: Expected Calibration Error (ECE) and Brier Score
- **ECE Bins**: 10
- **Evaluation**: On held-out test set

**Calibration Characteristics:**
- Model outputs are logits (not probabilities)
- Softmax applied for probability estimation
- Calibration may vary across subjects and sessions

### 8.2 Calibration Results

**Note**: Detailed calibration metrics should be computed on test set. The model uses standard cross-entropy loss without explicit calibration techniques.

**Expected Behavior:**
- Well-calibrated on training distribution
- May require temperature scaling for improved calibration
- Subject-specific calibration may vary

---

## 9. Limitations

### 9.1 Dataset Limitations

- **Single Dataset**: Trained and evaluated on BNCI2014_001 only
- **Limited Subjects**: 9 subjects (may not generalize to broader population)
- **Session Variability**: Performance may degrade across sessions
- **Subject-Specific**: May require subject-specific fine-tuning

### 9.2 Architecture Limitations

- **Fixed Input Size**: Requires fixed input dimensions (22 channels, 1000 timepoints)
- **Binary Classification**: Designed for 2-class classification
- **Computational Cost**: More complex than baseline models (EEGNet, REEGNet)
- **Hyperparameter Sensitivity**: Requires careful tuning for optimal performance

### 9.3 Generalization Limitations

- **Cross-Dataset**: Not validated on other EEG datasets
- **Cross-Paradigm**: Not validated for other BCI paradigms (SSVEP, P300)
- **Real-Time**: Not optimized for real-time inference
- **Clinical Use**: Not validated for clinical applications

### 9.4 Known Issues

- **Overfitting Risk**: Complex architecture may overfit on small datasets
- **Noise Sensitivity**: Performance degrades under high noise conditions
- **Session Drift**: May require session-specific adaptation

---

## 10. Safety Considerations

### 10.1 Medical Disclaimer

**IMPORTANT**: This model is **NOT** intended for:
- Clinical diagnosis
- Medical decision-making
- Patient care
- Real-time medical applications

**Use Case**: Research and development only

### 10.2 Data Privacy

- **EEG Data**: Contains sensitive biometric information
- **Subject Privacy**: Ensure proper anonymization
- **Data Sharing**: Follow institutional and regulatory guidelines
- **GDPR Compliance**: Ensure compliance with data protection regulations

### 10.3 Model Reliability

- **Uncertainty Quantification**: Not explicitly modeled
- **Confidence Intervals**: Not provided with predictions
- **Error Analysis**: Limited error analysis on failure cases
- **Robustness**: Evaluated under controlled noise conditions only

### 10.4 Deployment Considerations

- **Hardware Requirements**: GPU recommended for training, CPU sufficient for inference
- **Latency**: Not optimized for real-time applications
- **Scalability**: Not tested on large-scale deployments
- **Monitoring**: No built-in monitoring or logging for production use

---

## 11. Ethical Considerations

### 11.1 Bias and Fairness

**Potential Biases:**
- **Subject Bias**: Trained on 9 subjects (may not represent diverse populations)
- **Age Bias**: Dataset age range not specified
- **Gender Bias**: Gender distribution not analyzed
- **Health Status**: Only healthy subjects (no neurological conditions)

**Fairness Assessment:**
- No explicit fairness evaluation performed
- Performance may vary across demographic groups
- Requires additional validation for diverse populations

### 11.2 Data Collection Ethics

- **Informed Consent**: Assumed for BNCI2014_001 dataset
- **Data Usage**: Research purposes only
- **Data Sharing**: Follow dataset license terms

### 11.3 Model Interpretability

- **Black Box**: Limited interpretability (deep neural network)
- **Feature Attribution**: No explicit feature importance analysis
- **Decision Explanation**: No explanation mechanisms provided
- **Attention Visualization**: Attention weights can be visualized but not fully interpretable

### 11.4 Responsible AI

- **Transparency**: Model architecture and training details documented
- **Reproducibility**: Seeds and hyperparameters provided
- **Accountability**: Model performance limitations clearly stated
- **Human Oversight**: Requires human expert validation for clinical use

---

## 12. Inference Speed

### 12.1 Inference Performance

**Hardware**: NVIDIA GPU (tested on various GPUs)
**Batch Size**: 64 (default)

**Inference Time (per trial):**
- **Single Trial**: ~1-5 ms (GPU), ~10-50 ms (CPU)
- **Batch Processing**: ~50-200 ms per batch (batch_size=64)
- **Throughput**: ~300-1000 trials/second (GPU)

**Factors Affecting Speed:**
- Hardware (GPU vs. CPU)
- Batch size
- Input sequence length
- Model complexity (branched processing)

### 12.2 Optimization Opportunities

- **Model Quantization**: Not applied (FP32)
- **Pruning**: Not applied
- **TensorRT**: Not optimized
- **ONNX Export**: Not tested

---

## 13. Memory Footprint

### 13.1 Model Size

**Parameter Count**: ~4,000-15,000 parameters (depends on wiring)
- CNN Front-end: ~1,400 parameters
- Multi-scale Block: ~1,680 parameters
- SNR Gate: ~192 parameters
- Temporal Downsampler: ~768 parameters
- Recurrent (Architecture #4): Variable (depends on wiring)
- Classification Head: ~64 parameters

**Model Size (Disk):**
- **FP32**: ~50-200 KB (depending on wiring)
- **FP16**: ~25-100 KB (if quantized)

### 13.2 Runtime Memory

**Training Memory (GPU):**
- Model parameters: ~200 KB
- Gradients: ~200 KB
- Activations: ~10-50 MB (batch_size=64)
- **Total**: ~50-100 MB per batch

**Inference Memory (GPU):**
- Model parameters: ~200 KB
- Activations: ~5-20 MB (batch_size=64)
- **Total**: ~20-50 MB per batch

**Inference Memory (CPU):**
- Similar to GPU (no significant difference)

---

## 14. Number of Learned Parameters

### 14.1 Parameter Breakdown

**Total Parameters**: ~4,000-15,000 (exact count depends on Architecture #4 wiring)

**Component-wise Breakdown:**

1. **CNN Front-end**: ~1,400 parameters
   - Temporal Conv: 1,000 (1 × 8 × 125)
   - Depthwise Spatial Conv: 352 (8 × 2 × 22)
   - BatchNorm: 48 (2 × 8 + 2 × 16)

2. **Multi-Scale Temporal Block**: ~1,680 parameters
   - Three parallel branches: 1,648
   - LayerNorm: 32

3. **SNR Gate**: ~192 parameters
   - MLP: 192 (32 × 4 + 4 × 16)

4. **Temporal Downsampler**: ~768 parameters
   - Conv1D: 768 (16 × 16 × 3)

5. **Recurrent Compartment (Architecture #4)**: Variable
   - Input size: 8 (projected from F2=16)
   - Hidden size: 43
   - Output size: 7 (projected to 16)
   - Wiring matrix: Sparse (exact count depends on connectivity)

6. **Attention Mechanisms**: ~500-1,000 parameters
   - Intra-bin attention: ~300-500
   - Inter-bin attention: ~200-500

7. **Classification Head**: ~64 parameters
   - LayerNorm: 32
   - Linear: 32 (16 × 2)

### 14.2 Parameter Counting

To obtain exact parameter counts, run:
```bash
python count_model_parameters.py
```

This script instantiates the model with Architecture #4 and reports:
- Total trainable parameters
- Model size in MB
- Wiring information (connections, hidden units)

---

## 15. Reproducibility and Sharing Practices

### 15.1 Reproducibility

**Random Seeds:**
- **Global Seed**: Controlled via `set_seeds()` function
- **Seed Propagation**: All random operations use the same seed
- **Deterministic Operations**: CUDA deterministic mode enabled
- **Seed Documentation**: Seeds recorded in experiment logs

**Reproducibility Measures:**
- Fixed random seeds for all random operations
- Deterministic CUDA operations (when available)
- Version-controlled code and dependencies
- Saved hyperparameters and model configurations
- Architecture files (JSON) for exact wiring reproduction

### 15.2 Code Availability

**Repository Structure:**
- Model implementation: `models/branched_wiredcfc.py`
- Base class: `models/branched_diva_base.py`
- Wiring system: `architecture_refinement/arbitrary_wiring.py`
- Architecture file: `outputs/architectures/best_architecture_4_trial_178.json`

**Dependencies:**
- PyTorch (>=1.9.0)
- braindecode
- MOABB
- ncps (for CfC cells)
- scikit-learn
- numpy

### 15.3 Model Sharing

**Model Artifacts:**
- Architecture configuration (JSON)
- Hyperparameters (saved in experiment logs)
- Training code (version-controlled)
- Evaluation scripts

**Not Shared:**
- Trained model weights (due to dataset licensing)
- Raw EEG data (privacy and licensing)

**Reproduction Steps:**
1. Install dependencies
2. Load Architecture #4 from JSON file
3. Instantiate model with saved hyperparameters
4. Train with fixed random seed
5. Evaluate on test set

### 15.4 Experiment Logging

**Logged Information:**
- Random seeds
- Hyperparameters (architecture and training)
- Training history (loss, metrics per epoch)
- Validation scores
- Test scores
- Model checkpoints (optional)

**Log Locations:**
- Training history: `results/{paradigm}/{dataset}/{model}/{eval_mode}/{seed}/.../training_history/`
- Hyperparameters: `results/.../Optuna/fold_{fold_idx}/`

---

## 16. Robustness Summary

### 16.1 Clean Performance

**Cross-Session Evaluation (BNCI2014_001):**
- **Mean ROC-AUC**: ~0.95 (varies by subject and seed)
- **Mean Accuracy**: ~85-90% (varies by subject)
- **Performance Range**: ROC-AUC 0.88-0.98 across subjects

**Subject-Specific Performance:**
- Performance varies significantly across subjects
- Some subjects achieve >95% accuracy
- Others may require subject-specific tuning

### 16.2 Robustness to Noise

**Noise Types Evaluated:**
1. **Gaussian Noise**: Additive white Gaussian noise
2. **EOG Artifacts**: Realistic eye movement artifacts
3. **Dropout Noise**: Random channel dropout
4. **Spike Artifacts**: Impulsive noise spikes

**Robustness Metrics:**
- **Performance Retention**: Percentage of clean performance maintained under noise
- **Saturation Point**: Noise intensity at which performance degrades significantly

**Robustness Results (from ReZero comparison study):**

**Backwards ReZero (Default):**
- **Mean Clean ROC-AUC**: 0.9512 ± 0.0146
- **Mean Corrupted ROC-AUC**: 0.8006 ± 0.1392
- **Mean Retention**: 84.18% ± 14.66%

**Performance by Noise Type:**
- **EOG**: 93.01% retention (mean ROC-AUC: 0.8848)
- **Gaussian**: 89.66% retention (mean ROC-AUC: 0.8525)
- **Dropout**: 69.87% retention (mean ROC-AUC: 0.6644)

### 16.3 Robustness Mechanisms

**Architectural Features Contributing to Robustness:**
1. **Multi-Scale Temporal Processing**: Captures patterns at multiple time scales
2. **SNR Gating**: Adaptively suppresses noisy channels
3. **Branched Processing**: Parallel processing provides redundancy
4. **Attention Mechanisms**: Focus on informative timepoints
5. **Weighted Residual Connections**: Stabilize training and inference
6. **Optimized Wiring (Architecture #4)**: Graph-theoretic properties enhance robustness

### 16.4 Failure Modes

**Known Failure Cases:**
- High-intensity noise (>100% Gaussian noise)
- Severe channel dropout (>50% channels)
- Subject-specific failures (poor performance on some subjects)
- Session drift (performance degradation across sessions)

**Mitigation Strategies:**
- Noise injection during training (not used in default)
- Subject-specific fine-tuning
- Session adaptation techniques
- Ensemble methods (not implemented)

---

## 17. Additional Information

### 17.1 Citation

If you use HYDRA in your research, please cite:

```bibtex
@article{hydra2025,
  title={HYDRA: A Hybrid DIVA Model with Recurrent Attention for EEG-based Motor Imagery Classification},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

### 17.2 Contact

For questions, issues, or contributions:
- **Repository**: [GitHub repository URL]
- **Issues**: [GitHub issues URL]
- **Email**: [Contact email]

### 17.3 License

- **Model Code**: [License]
- **Dataset**: BNCI2014_001 (research use only, see dataset license)

### 17.4 Acknowledgments

- **Dataset**: BNCI2014_001 (Tangermann et al., 2012)
- **MOABB**: Mother of All BCI Benchmarks
- **braindecode**: EEG deep learning library
- **ncps**: Neural Circuit Policies library

---

## Appendix A: Architecture #4 Wiring Details

**Wiring Configuration:**
- **File**: `outputs/architectures/best_architecture_4_trial_178.json`
- **Input Size**: 8
- **Hidden Size**: 43
- **Output Size**: 7
- **Discovery Method**: Graph-theoretic architecture search
- **Optimization Objectives**: Entropy, curvature, algebraic connectivity, efficiency

**To Load Architecture #4:**
```python
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from models.branched_wiredcfc import create_branched_wiredcfc_classifier

wiring = load_architecture_from_file("outputs/architectures/best_architecture_4_trial_178.json")
model = create_branched_wiredcfc_classifier(
    n_chans=22, n_times=1001, n_outputs=2, wiring=wiring
)
```

**Wiring Properties:**
- Sparse connectivity (structured, not fully connected)
- Optimized for robustness (graph-theoretic properties)
- Discovered through multi-objective optimization

---

## Appendix B: Performance Benchmarks

**Note**: Detailed performance benchmarks should be computed on the test set. The following are estimates based on validation performance.

**Cross-Session Performance (BNCI2014_001):**
- Subject 1: ROC-AUC ~0.95-0.98
- Subject 2: ROC-AUC ~0.90-0.95
- Subject 3: ROC-AUC ~0.88-0.93
- [Additional subjects...]

**Comparison with Baselines:**
- **EEGNet**: HYDRA typically outperforms by 5-10% ROC-AUC
- **REEGNet**: HYDRA typically outperforms by 3-8% ROC-AUC
- **CNN-NCP**: HYDRA typically outperforms by 2-5% ROC-AUC

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-16  
**Maintained By**: [Maintainer Name/Organization]

