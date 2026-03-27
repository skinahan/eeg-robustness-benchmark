# CfC Branching Inference Benchmark Specification

## Purpose and Hypothesis

**Objective:** Quantify the relative inference speed of the CfC recurrent compartment when using **branched** vs **unbranched** temporal processing in the BranchedWiredCfC (HYDRA) architecture.

**Hypothesis:** Branching reduces effective sequence length per CfC call (L=48 vs T2~125), lowering time-complexity from O(T2) to O(L) per path. Theoretically, branched inference should be faster.

---

## Experimental Protocol

### Data
- **Dataset:** BNCI2014_001 (MOABB)
- **Subject:** 1 only
- **Preprocessing:** `config.get_paradigm(dataset="BNCI2014_001")` and `paradigm.get_data(dataset, subjects=[1])`

### Seeds
- **3 fixed seeds:** [42, 43, 44]
- For each seed: set global RNG seeds, initialize model, run benchmark
- Report median latency and throughput per seed; aggregate mean ± std across seeds

### Models Compared
| Variant      | Implementation                         | CfC input shape | Effective sequence length |
|-------------|----------------------------------------|-----------------|---------------------------|
| **Branched** | BranchedWiredCfC (bin_len=48, bin_stride=44) | [B×NB, L, F2]   | L (48)                    |
| **Unbranched** | BranchedWiredCfC (bin_len=512, bin_stride=512) | [B, T2, F2]   | T2 (~125)                 |

Both use BranchedWiredCfC for fair comparison (same wiring/projection handling). Unbranched achieves a single bin via large bin_len/stride so unfold yields one window.

### Metrics (ML Conference Standards)
- **Primary:** Median latency per batch (ms)
- **Derived:** Per-sample latency (ms), throughput (samples/sec)
- **Benchmark protocol:** 50 warm-up passes, 200 timed trials, `torch.cuda.synchronize()` on GPU

---

## Fair Comparison Criteria

1. Same wiring (load architecture per model; BranchedWiredCfC uses projection layers when dims mismatch)
2. Same hyperparameters: F1=8, D=2, bin_len=48, bin_stride=44, etc.
3. Same input: identical BNCI2014_001 subject 1 preprocessed epochs
4. Same batch size (default 32)
5. Same device and precision: `model.eval()`, `torch.no_grad()`

---

## Unified Experiment Runner Extrapolation

The benchmark extrapolates per-batch latency to estimate **inference time savings** for a typical unified experiment runner run. This is an *estimate*; training is dominated by backward passes; inference speedup mainly affects validation and test phases.

### Default Parameters (from experiment_config.yaml, globals.py)

| Parameter            | Value | Source                          |
|---------------------|-------|---------------------------------|
| n_models            | 1     | Single model comparison         |
| n_seeds             | 5     | experiment_config.yaml          |
| n_subjects          | 9     | BNCI2014_001                    |
| max_epochs          | 200   | globals.get_max_epochs_for_dataset |
| batches_per_epoch   | 5     | Conservative estimate (train+val) |
| test_perturb_batches | 100  | 20 steps × 4 noise types × ~1.25 batches |

### Formula
```
total_inference_batches = n_models × n_seeds × n_subjects × (max_epochs × batches_per_epoch + test_perturb_batches)
unbranched_time_sec = total_batches × unbranched_median_ms / 1000
branched_time_sec = total_batches × branched_median_ms / 1000
time_saved_sec = unbranched_time_sec - branched_time_sec
```

### Output
The script reports:
- Total inference batches
- Unbranched and branched inference time (seconds and hours)
- Time saved (seconds, hours, and percentage)

---

## How to Run

```bash
# From project root
python experiments/benchmark_cfc_branching_inference.py

# With options
python experiments/benchmark_cfc_branching_inference.py \
  --architecture outputs/architectures/best_architecture_4_trial_178.json \
  --batch-size 32 \
  --n-warmup 50 \
  --n-trials 200 \
  --seeds 42,43,44
```

**Requirements:** MOABB, BNCI2014_001 downloaded, architecture file at `outputs/architectures/best_architecture_4_trial_178.json`

---

## Interpretation

- **Speedup > 1.0:** Branched is faster (e.g., 1.5x means branched takes ~67% of unbranched time)
- **Theoretical expectation:** With L=48 vs T2~125, sequential CfC work ratio is ~2.6× less per path; actual speedup may be lower due to front-end overhead and parallelization
- **Unified runner estimate:** Use as a rough guide for experiment planning; actual savings depend on data splits, early stopping, and hardware
