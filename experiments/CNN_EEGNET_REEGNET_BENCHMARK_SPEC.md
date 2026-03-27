# CNN-NCP, EEGNet, REEGNet Inference Benchmark Specification

## Purpose

Compares inference speed of three EEG classification models used in the unified experiment runner:
- **CNN-NCP** (CNNNCPv3): CNN + NCP recurrent backbone
- **EEGNet** (EEGNetv4): Braindecode EEGNet
- **REEGNet**: REEGNet with LSTM-based temporal processing

Uses the **exact same setup** as the CfC branching benchmark for consistency.

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

### Metrics (ML Conference Standards)
- **Primary:** Median latency per batch (ms)
- **Derived:** Throughput (samples/sec)
- **Protocol:** 50 warm-up passes, 200 timed trials, `torch.cuda.synchronize()` on GPU

---

## Fair Comparison Criteria

1. Same input: identical BNCI2014_001 subject 1 preprocessed epochs
2. Same batch size (32)
3. Same device and precision: `model.eval()`, `torch.no_grad()`
4. Each model uses its default factory (create_*_classifier) with no tuning

---

## Unified Experiment Runner Extrapolation

Uses the same parameters as the CfC benchmark:
- 1 model, 5 seeds, 9 subjects
- 200 max epochs, 5 batches/epoch, 100 test_perturb batches
- Formula: `total_batches × median_latency_ms / 1000` seconds

---

## How to Run

```bash
# From project root
python experiments/benchmark_cnn_eegnet_reegnet_inference.py

# With options
python experiments/benchmark_cnn_eegnet_reegnet_inference.py \
  --batch-size 32 \
  --n-warmup 50 \
  --n-trials 200 \
  --seeds 42,43,44 \
  --models cnn_ncp,eegnet,reegnet
```

**Requirements:** MOABB, BNCI2014_001 downloaded
