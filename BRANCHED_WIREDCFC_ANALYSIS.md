# BranchedWiredCfC: A Comprehensive Architecture Analysis

## Abstract

This document provides a detailed scientific analysis of the BranchedWiredCfC model architecture, a hybrid neural network designed for EEG signal classification. The model combines a CNN-based feature extraction front-end with a branched recurrent processing module using Closed-form Continuous-time (CfC) cells with arbitrary wiring configurations. This analysis examines the architectural components, design choices, parameter specifications, and comparative analysis with baseline models.

---

## 1. Architecture Overview

The BranchedWiredCfC model is a hierarchical architecture that processes EEG signals through six distinct stages:

1. **CNN Feature Extraction Front-end**: Temporal and spatial convolution layers
2. **Multi-scale Temporal Integration**: Dilated convolutional blocks for noise-stable temporal processing
3. **Signal-to-Noise Ratio (SNR) Gating**: Adaptive channel suppression
4. **Temporal Downsampling**: Sequence length reduction via 1D convolution
5. **Branched Recurrent Processing**: Parallel CfC processing over temporal bins with arbitrary wiring
6. **Classification Head**: Attention-based pooling and final classification

The model accepts input tensors of shape `(B, C, T)` where `B` is batch size, `C` is the number of EEG channels, and `T` is the number of timepoints. The output is a logits tensor of shape `(B, n_outputs)` for classification.

---

## 2. CNN Feature Extraction Front-end

### 2.1 Comparison with EEGNet and REEGNet

The CNN front-end of BranchedWiredCfC shares structural similarities with EEGNet and REEGNet but incorporates several key differences:

#### 2.1.1 Temporal Convolution Layer

**BranchedWiredCfC:**
- **Layer**: `Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, kernel_length), stride=(1, 1), padding=(0, kernel_length // 2), bias=False)`
- **Default Parameters**: `F1=8`, `kernel_length=125`
- **Activation**: ELU (Exponential Linear Unit)
- **Normalization**: BatchNorm2d with `momentum=0.01`, `eps=1e-3`
- **Optional**: Spectral normalization (disabled by default)

**EEGNet (EEGNetv4 from braindecode):**
- Uses a similar temporal convolution structure
- Typically employs `F1=8` filters
- Uses separable convolution blocks
- Kernel size typically smaller (around 64-128 samples)

**REEGNet:**
- **Layer**: `Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)`
- Fixed `F1=8` filters
- Fixed kernel size of 15 timepoints
- Uses ELU activation
- Standard BatchNorm2d

**Key Differences:**
1. **Kernel Length**: BranchedWiredCfC uses a significantly larger default kernel length (125) compared to REEGNet (15), allowing for longer temporal context capture. This is configurable, unlike REEGNet's fixed value.
2. **Parameterization**: BranchedWiredCfC uses fully parameterized `F1` and `kernel_length`, while REEGNet uses hardcoded values.
3. **Spectral Normalization**: BranchedWiredCfC optionally supports spectral normalization on the first convolution layer for improved noise robustness (disabled by default).

#### 2.1.2 Depthwise Spatial Convolution

**BranchedWiredCfC:**
- **Layer**: `Conv2d(in_channels=F1, out_channels=F2, kernel_size=(n_chans, 1), stride=(1, 1), padding=(0, 0), groups=F1, bias=False)`
- **Default Parameters**: `F1=8`, `D=2`, resulting in `F2=16` output channels
- **Operation**: Depthwise separable convolution across spatial (channel) dimension
- **Purpose**: Learns spatial filters that create "virtual sensors" by combining information across EEG channels

**REEGNet:**
- **Layer**: `Conv2d(in_channels=8, out_channels=16, kernel_size=(n_chans, 1), groups=8, stride=(1, 1), padding=(0, 0), bias=False)`
- Fixed `F1=8`, `F2=16` (equivalent to `D=2`)
- Same depthwise operation but with hardcoded dimensions

**Key Differences:**
1. **Flexibility**: BranchedWiredCfC allows configurable `F1` and `D` parameters, enabling architectural search and optimization.
2. **Architecture**: Both use identical depthwise convolution structure, but BranchedWiredCfC's parameterization enables hyperparameter tuning.

#### 2.1.3 Temporal Pooling

**BranchedWiredCfC:**
- **Layer**: `AvgPool2d(kernel_size=(1, pool_time), stride=(1, pool_time))`
- **Default Parameter**: `pool_time=4`
- **Purpose**: Anti-aliasing temporal downsampling to reduce computational load and prevent aliasing artifacts

**REEGNet:**
- **Layer**: `AvgPool2d(kernel_size=(1, 4), stride=(1, 4))`
- Fixed pooling factor of 4

**Key Differences:**
1. **Configurability**: BranchedWiredCfC's `pool_time` is a hyperparameter, while REEGNet uses a fixed value.

#### 2.1.4 Dropout Regularization

**BranchedWiredCfC:**
- **Default**: `drop_prob=0.25`
- Applied after temporal pooling

**REEGNet:**
- **Default**: `drop_prob=0.15`
- Applied after temporal pooling

**Key Differences:**
1. **Regularization Strength**: BranchedWiredCfC uses higher default dropout (0.25 vs 0.15), suggesting a focus on preventing overfitting in the more complex architecture.

---

## 3. Multi-scale Temporal Integration

### 3.1 Architecture

BranchedWiredCfC incorporates a **Multi-Scale Temporal Block** (`_MultiScaleTemporalBlock1D`) that is not present in standard EEGNet or REEGNet architectures. This block performs noise-stable temporal integration through parallel dilated convolutional branches.

**Structure:**
- **Input**: `(B, F2, T1)` - Features after CNN front-end and pooling
- **Operation**: Three parallel depthwise-separable Conv1D branches with different dilations
- **Default Parameters**:
  - Kernels: `(9, 15, 31)`
  - Dilations: `(1, 4, 16)`
- **Processing**:
  1. Each branch applies depthwise-separable convolution with its respective kernel size and dilation
  2. Branches are summed element-wise
  3. Residual connection adds the original input
  4. Layer normalization applied over channel dimension
  5. ELU activation
- **Output**: `(B, F2, T1)` - Same temporal length, enhanced features

**Purpose:**
- Captures temporal patterns at multiple scales simultaneously
- Dilated convolutions increase receptive field without increasing parameters
- Residual connection preserves original signal information
- Layer normalization stabilizes training and improves noise robustness

**Comparison with Baselines:**
- **EEGNet**: Does not include multi-scale temporal processing
- **REEGNet**: Does not include multi-scale temporal processing
- This is a **unique architectural component** of BranchedWiredCfC that enhances temporal feature extraction.

---

## 4. Signal-to-Noise Ratio (SNR) Gate

### 4.1 Architecture

The **SNR Gate** (`_SNRGate`) is another component not found in standard EEGNet or REEGNet. It implements Wiener-like shrinkage to suppress noisy channels.

**Structure:**
- **Input**: `(B, F2, T1)`
- **Operation**:
  1. Computes per-channel statistics: mean and log-variance across time dimension
  2. Concatenates statistics: `(B, 2*F2)`
  3. Passes through MLP: `Linear(2*F2 → F2/reduction) → ReLU → Linear(F2/reduction → F2) → Sigmoid`
  4. Generates per-channel gain factors `γ ∈ [0, 1]`
  5. Applies element-wise multiplication: `x_gated = x * γ`
- **Default Parameters**: `snr_reduction=4` (reduction factor for hidden layer)
- **Output**: `(B, F2, T1)` - Gated features with suppressed noisy channels

**Purpose:**
- Adaptively suppresses channels with low signal-to-noise ratio
- Implements a form of attention mechanism based on signal quality
- Improves robustness to channel-specific noise and artifacts

**Comparison with Baselines:**
- **EEGNet**: No SNR gating mechanism
- **REEGNet**: No SNR gating mechanism
- This is a **unique architectural component** that enhances noise robustness.

---

## 5. Temporal Downsampling

### 5.1 Architecture

Before recurrent processing, BranchedWiredCfC applies an additional temporal downsampling step using 1D convolution.

**Layer**: `Conv1d(in_channels=F2, out_channels=F2, kernel_size=temporal_kernel_size, stride=temporal_stride, padding=temporal_kernel_size // 2, bias=False)`

**Default Parameters**:
- `temporal_kernel_size=3`
- `temporal_stride=2`

**Purpose**:
- Further reduces sequence length for efficient recurrent processing
- Maintains feature dimensionality while reducing temporal resolution
- Prepares features for binning and parallel processing

**Comparison with Baselines:**
- **EEGNet**: Uses separable convolution blocks for temporal processing, but not a dedicated downsampling layer
- **REEGNet**: Does not include this explicit temporal downsampling step before recurrent processing

---

## 6. Branched Recurrent Processing

### 6.1 Temporal Binning

The core innovation of BranchedWiredCfC is its **branched recurrent processing** strategy:

1. **Temporal Chunking**: The downsampled feature sequence is divided into overlapping temporal bins
   - **Default Parameters**: `bin_len=48`, `bin_stride=44`
   - **Operation**: Uses `unfold` to create overlapping windows
   - **Output Shape**: `(B, NB, L, F2)` where `NB` is number of bins, `L` is bin length

2. **Parallel Processing**: Bins are reshaped to `(B*NB, L, F2)` and processed in parallel through identical CfC cells

### 6.2 Closed-form Continuous-time (CfC) Cells

**Architecture:**
- **Cell Type**: CfC (Closed-form Continuous-time) with arbitrary wiring
- **Input Size**: `F2` (default: 16)
- **Output Size**: `recurrent_output_size` (default: `F2` for residual compatibility)
- **Wiring**: Arbitrary wiring configuration (e.g., Architecture 4 from optimization search)
- **Default CfC Parameters**:
  - `mixed_memory=True`: Enables mixed memory mechanism
  - `mode="default"`: Standard CfC operation mode
  - `activation="lecun_tanh"`: LeCun's tanh activation

**Arbitrary Wiring:**
- The wiring structure is defined by an `ArbitraryWiring` instance
- Architecture 4 (from `best_architecture_4_trial_178.json`) represents an optimized wiring configuration discovered through architecture search
- The wiring defines the connectivity pattern between input, hidden, and output units in the CfC cell
- This allows for sparse, structured connectivity patterns that can be optimized for specific tasks

**Processing:**
- Each bin is processed independently through the same CfC cell
- Returns sequences: `(B*NB, L, H)` where `H` is the recurrent output size

### 6.3 Weighted Residual Connections

After CfC processing, a weighted residual connection is applied:

**Operation**: `x_out = x_cfc * (1 - α) + x_residual * α`

Where:
- `α = weight_residual` is a learnable parameter (initialized to 0.0, using "backwards_rezero" strategy)
- At initialization (α=0): output = x_cfc (CfC/recurrent at full strength)
- This empirically validated approach outperforms standard ReZero (identity at init) for temporal modeling
- Allows the model to learn the optimal balance between CfC-processed features and original features

**Purpose**:
- Stabilizes training by providing gradient pathways through residual connections
- Allows the model to adaptively weight the contribution of recurrent processing
- Ensures compatibility with the original feature space

### 6.4 Intra-bin Attention Pooling

Within each bin, temporal attention pooling is applied:

**Architecture**: `TemporalAttnPool`
- **Input**: `(B*NB, L, H)`
- **Operation**:
  1. Learnable query vector: `q ∈ R^H`
  2. Key projection: `k = tanh(W_k * x)` where `W_k: H → H`
  3. Attention scores: `att = (k · q) / √H`
  4. Softmax normalization over time dimension
  5. Weighted sum: `z = Σ(att_i * x_i)`
- **Output**: `(B*NB, H)` - Per-bin summary vectors

**Purpose**:
- Aggregates temporal information within each bin
- Uses attention to focus on most informative timepoints
- Reduces sequence length while preserving important information

### 6.5 Inter-bin Fusion

After intra-bin pooling, bins are fused to restore global context:

**Options**:
1. **Attention Fusion** (default: `fusion="attn"`):
   - Uses another `TemporalAttnPool` to pool across bins
   - Allows the model to attend to most relevant bins
   
2. **Mean Fusion** (`fusion="mean"`):
   - Simple average pooling across bins
   - Faster but less expressive

**Output**: `(B, H)` - Global feature representation

**Comparison with Baselines:**
- **EEGNet**: Uses global average pooling directly after separable convolutions
- **REEGNet**: Uses LSTM for temporal processing, then global average pooling
- **CNN-NCP**: Uses single CfC/NCP cell over entire sequence, then global pooling
- **BranchedWiredCfC**: **Unique** in using parallel bin processing with attention-based fusion

---

## 7. Classification Head

### 7.1 Architecture

**Components**:
1. **Layer Normalization**: `LayerNorm(recurrent_output_size)`
2. **Dropout**: `Dropout(p=drop_prob)` with default `drop_prob=0.25`
3. **Linear Layer**: `Linear(recurrent_output_size, n_outputs)`

**Purpose**:
- Final feature normalization and regularization
- Maps from feature space to classification logits
- Lightweight design to prevent overfitting

---

## 8. Default Hyperparameters

### 8.1 Complete Parameter Specification

**CNN Front-end:**
- `F1=8`: Number of temporal filters
- `D=2`: Depthwise multiplier (F2 = F1 * D = 16)
- `kernel_length=125`: Temporal kernel size in first convolution
- `pool_time=4`: Temporal pooling factor
- `drop_prob=0.25`: Dropout probability
- `bn_momentum=0.01`: BatchNorm momentum
- `bn_eps=1e-3`: BatchNorm epsilon
- `use_spectral_norm_first_conv=False`: Spectral normalization (disabled)

**Multi-scale Temporal Block:**
- `ms_kernels=(9, 15, 31)`: Kernel sizes for parallel branches
- `ms_dilations=(1, 4, 16)`: Dilation factors for parallel branches

**Temporal Downsampling:**
- `temporal_kernel_size=3`: 1D convolution kernel size
- `temporal_stride=2`: Downsampling stride

**SNR Gate:**
- `snr_reduction=4`: Reduction factor for hidden layer

**Binning:**
- `bin_len=48`: Number of timesteps per bin (after downsampling)
- `bin_stride=44`: Step between bin starts (creates overlap)
- `fusion="attn"`: Fusion method ("attn" or "mean")

**CfC Parameters:**
- `mixed_memory=True`: Enable mixed memory mechanism
- `mode="default"`: CfC operation mode
- `activation="lecun_tanh"`: Activation function
- `recurrent_output_size=None`: Defaults to F2 (16) for residual compatibility

**Training Parameters (from `create_branched_wiredcfc_classifier`):**
- `optimizer=AdamW`
- `learning_rate=1e-2`
- `weight_decay=0`
- `batch_size=64`
- `gradient_clip_value=1.0`
- `learning_rate_scheduler=ExponentialLR` with `gamma=0.97`

---

## 9. Parameter Count Comparison

### 9.1 Methodology

To compare parameter counts across architectures, models were instantiated with standard input dimensions for BNCI2014_001 dataset:
- **Channels**: 22
- **Timepoints**: 1001
- **Outputs**: 2

### 9.2 Comparative Analysis

**Note**: Exact parameter counts should be verified using the provided `count_model_parameters.py` script. The following provides architectural insights into parameter scaling:

#### 9.2.1 CNN Front-end Parameters

**BranchedWiredCfC:**
- Temporal Conv: `1 * F1 * kernel_length = 1 * 8 * 125 = 1,000` parameters
- Depthwise Spatial Conv: `F1 * D * n_chans = 8 * 2 * 22 = 352` parameters
- BatchNorm parameters: `2 * F1 + 2 * F2 = 2 * 8 + 2 * 16 = 48` parameters
- **Subtotal**: ~1,400 parameters

**EEGNet:**
- Similar structure but typically uses smaller kernel sizes
- Estimated: ~800-1,200 parameters (depending on configuration)

**REEGNet:**
- Temporal Conv: `1 * 8 * 15 = 120` parameters
- Depthwise Spatial Conv: `8 * 2 * 22 = 352` parameters
- BatchNorm: `48` parameters
- **Subtotal**: ~520 parameters

**CNN-NCP (CNNNCPv3):**
- Temporal Conv: `1 * F1 * kernel_length = 1 * 8 * 128 = 1,024` parameters (default)
- Depthwise Spatial Conv: `8 * 2 * 22 = 352` parameters
- BatchNorm: `48` parameters
- Temporal Downsampler: `F2 * F2 * kernel_size = 16 * 16 * 3 = 768` parameters
- **Subtotal**: ~2,200 parameters

#### 9.2.2 Multi-scale Temporal Block Parameters

**BranchedWiredCfC:**
- Three parallel branches with depthwise-separable convolutions
- Each branch: `F2 * kernel + F2 * F2 = 16 * kernel + 256` parameters
- Total: `16 * (9 + 15 + 31) + 3 * 256 = 880 + 768 = 1,648` parameters
- LayerNorm: `2 * F2 = 32` parameters
- **Subtotal**: ~1,680 parameters

**EEGNet/REEGNet/CNN-NCP**: No equivalent component

#### 9.2.3 SNR Gate Parameters

**BranchedWiredCfC:**
- MLP: `(2*F2) * (F2/reduction) + (F2/reduction) * F2 = (32 * 4) + (4 * 16) = 128 + 64 = 192` parameters
- **Subtotal**: ~192 parameters

**EEGNet/REEGNet/CNN-NCP**: No equivalent component

#### 9.2.4 Recurrent Processing Parameters

**BranchedWiredCfC (Architecture 4):**
- Wiring structure defined by Architecture 4 (from optimization search)
- Parameter count depends on wiring sparsity and structure
- CfC cell parameters scale with:
  - Input size: `F2 = 16`
  - Hidden units: Defined by wiring
  - Output size: `F2 = 16` (for residual compatibility)
- **Note**: Exact count requires loading Architecture 4 wiring and counting connections

**CNN-NCP:**
- AutoNCP wiring with configurable sparsity
- Default: `ncp_hidden_dim=32`, `sparsity=0.85`
- Parameters scale with wiring structure and sparsity

**REEGNet:**
- LSTM: `4 * (input_size * hidden_size + hidden_size^2 + hidden_size)`
- With `input_size=4`, `hidden_size=32`: `4 * (4*32 + 32*32 + 32) = 4 * 1,280 = 5,120` parameters
- **Subtotal**: ~5,120 parameters

#### 9.2.5 Classification Head Parameters

**BranchedWiredCfC:**
- LayerNorm: `2 * H = 2 * 16 = 32` parameters
- Linear: `H * n_outputs = 16 * 2 = 32` parameters
- **Subtotal**: ~64 parameters

**EEGNet/REEGNet/CNN-NCP:**
- Similar lightweight classification heads
- Estimated: ~30-100 parameters

### 9.3 Total Parameter Estimates

**BranchedWiredCfC (Architecture 4):**
- CNN Front-end: ~1,400
- Multi-scale Block: ~1,680
- SNR Gate: ~192
- Temporal Downsampler: ~768
- Recurrent (Architecture 4): *Depends on wiring*
- Classification Head: ~64
- **Estimated Total**: ~4,000-15,000 parameters (depending on wiring)

**EEGNet:**
- Estimated: ~2,000-5,000 parameters

**REEGNet:**
- CNN Front-end: ~520
- LSTM: ~5,120
- Separable Conv: ~1,000
- Classification Head: ~50
- **Estimated Total**: ~6,700 parameters

**CNN-NCP:**
- CNN Front-end: ~2,200
- Temporal Downsampler: ~768
- NCP/CfC: *Depends on wiring and sparsity*
- Separable Conv: ~1,000
- Classification Head: ~50
- **Estimated Total**: ~4,000-10,000 parameters (depending on NCP configuration)

**Note**: These are rough estimates. For precise counts, run `count_model_parameters.py` with the actual model instances.

---

## 10. Architectural Innovations

### 10.1 Key Differentiators from Baselines

1. **Multi-scale Temporal Processing**: Unique parallel dilated convolution branches for multi-scale temporal feature extraction
2. **SNR Gating**: Adaptive channel suppression based on signal quality
3. **Branched Recurrent Processing**: Parallel processing of temporal bins with attention-based fusion
4. **Arbitrary Wiring**: Optimizable CfC wiring structure (e.g., Architecture 4) discovered through architecture search
5. **Weighted Residual Connections**: Learnable balance between recurrent and original features
6. **Hierarchical Attention**: Both intra-bin and inter-bin attention mechanisms

### 10.2 Design Rationale

1. **Noise Robustness**: Multi-scale blocks and SNR gating address EEG signal noise and artifacts
2. **Temporal Modeling**: Branched processing with attention captures both local (within-bin) and global (across-bin) temporal patterns
3. **Efficiency**: Parallel bin processing enables efficient computation while maintaining temporal context
4. **Flexibility**: Parameterized architecture allows for hyperparameter optimization and task-specific tuning

---

## 11. Implementation Details

### 11.1 Weight Initialization

All convolutional and linear layers use **Xavier (Glorot) uniform initialization** with `gain=1.0`. BatchNorm layers initialize weights to 1.0 and biases to 0.0.

### 11.2 Residual Connection Initialization

The weighted residual parameter `weight_residual` is initialized to **0.0** with the "backwards_rezero" strategy (default). 
This means the recurrent compartment starts at full strength rather than identity mapping. This approach has been 
empirically validated to provide ~6.4% better clean performance compared to standard ReZero (identity at init). 
See `REZERO_BACKWARDS_ANALYSIS.md` for theoretical justification.

### 11.3 Forward Pass Flow

```
Input (B, C, T)
  ↓
[1] Temporal Conv2D + BN + ELU → (B, F1, C, T)
  ↓
[2] Depthwise Spatial Conv2D + BN + ELU → (B, F2, 1, T)
  ↓
[3] AvgPool2D → (B, F2, 1, T1)
  ↓
[4] Dropout
  ↓
[5] Squeeze spatial dim → (B, F2, T1)
  ↓
[6] Multi-scale Temporal Block → (B, F2, T1)
  ↓
[7] SNR Gate → (B, F2, T1)
  ↓
[8] Temporal Downsampler (Conv1D) → (B, F2, T2)
  ↓
[9] Transpose → (B, T2, F2)
  ↓
[10] Chunk into bins → (B, NB, L, F2)
  ↓
[11] Reshape for parallel processing → (B*NB, L, F2)
  ↓
[12] CfC processing → (B*NB, L, H)
  ↓
[13] Weighted residual connection → (B*NB, L, H)
  ↓
[14] Intra-bin attention pooling → (B*NB, H)
  ↓
[15] Reshape → (B, NB, H)
  ↓
[16] Inter-bin fusion (attention or mean) → (B, H)
  ↓
[17] LayerNorm + Dropout
  ↓
[18] Linear classification → (B, n_outputs)
```

---

## 12. Comparison Summary

### 12.1 Architecture Component Comparison

| Component | BranchedWiredCfC | EEGNet | REEGNet | CNN-NCP |
|-----------|------------------|--------|---------|---------|
| Temporal Conv | ✓ (F1=8, kernel=125) | ✓ | ✓ (F1=8, kernel=15) | ✓ (F1=8, kernel=128) |
| Depthwise Spatial Conv | ✓ (F2=16) | ✓ | ✓ (F2=16) | ✓ (F2=16) |
| Temporal Pooling | ✓ (pool=4) | ✓ | ✓ (pool=4) | ✓ |
| Multi-scale Block | ✓ | ✗ | ✗ | ✗ |
| SNR Gate | ✓ | ✗ | ✗ | ✗ |
| Temporal Downsampler | ✓ (Conv1D) | ✗ | ✗ | ✓ (Conv1D) |
| Recurrent Processing | CfC (branched) | ✗ | LSTM | CfC/NCP (sequential) |
| Attention Pooling | ✓ (intra & inter-bin) | ✗ | ✗ | ✗ |
| Residual Connections | ✓ (weighted) | ✗ | ✗ | ✗ (some variants) |

### 12.2 Key Advantages

1. **Enhanced Temporal Modeling**: Multi-scale blocks and branched processing capture complex temporal patterns
2. **Noise Robustness**: SNR gating and multi-scale processing improve performance on noisy EEG signals
3. **Flexibility**: Parameterized architecture enables optimization and adaptation
4. **Efficiency**: Parallel bin processing maintains temporal context while enabling efficient computation

### 12.3 Trade-offs

1. **Complexity**: More components than baseline models, requiring careful hyperparameter tuning
2. **Parameter Count**: Additional components (multi-scale block, SNR gate) increase parameter count
3. **Computational Cost**: Branched processing and attention mechanisms add computational overhead

---

## 13. Conclusion

The BranchedWiredCfC model represents a sophisticated approach to EEG signal classification, combining proven CNN feature extraction with innovative recurrent processing strategies. Its unique components—multi-scale temporal integration, SNR gating, and branched attention-based processing—address key challenges in EEG analysis: noise robustness, temporal pattern recognition, and efficient computation.

The architecture's parameterization and use of optimizable wiring structures (e.g., Architecture 4) enable both architectural search and task-specific optimization, making it a flexible framework for EEG classification tasks.

---

## 14. References and Implementation Notes

- **Implementation**: `models/branched_wiredcfc.py`
- **Base Class**: `models/branched_diva_base.py`
- **Wiring System**: `architecture_refinement/arbitrary_wiring.py`
- **Architecture 4**: `outputs/architectures/best_architecture_4_trial_178.json`
- **Parameter Counting Script**: `count_model_parameters.py`

---

## Appendix A: Architecture 4 Wiring Details

Architecture 4 represents an optimized wiring configuration discovered through architecture search. The wiring structure defines the connectivity pattern for the CfC recurrent cells. To obtain detailed wiring information:

```python
from architecture_refinement.arbitrary_wiring import load_architecture_from_file
from models.branched_wiredcfc import create_branched_wiredcfc_classifier

wiring = load_architecture_from_file("outputs/architectures/best_architecture_4_trial_178.json")
model = create_branched_wiredcfc_classifier(n_chans=22, n_times=1001, n_outputs=2, wiring=wiring)
model.initialize()
wiring_info = model.module_.get_wiring_info()
print(wiring_info)
```

---

## Appendix B: Parameter Count Verification

To verify parameter counts, run:

```bash
python count_model_parameters.py
```

This script will:
1. Instantiate all models with standard parameters
2. Count trainable parameters
3. Compare parameter counts across architectures
4. Display a summary table

---

*Document generated for scientific research community analysis*
