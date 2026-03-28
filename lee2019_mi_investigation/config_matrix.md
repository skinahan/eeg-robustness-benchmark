# Lee2019_MI: configuration matrix (MOABB vs this repo)

This table audits **preprocessing and evaluation** choices. Values refer to [`config.py`](../config.py) and [`evaluation/unified_experiment_runner.py`](../evaluation/unified_experiment_runner.py) unless noted.

| Setting | MOABB default (dataset + `MotorImagery` paradigm) | This repo (`Lee2019_MI`) | Notes |
|---------|---------------------------------------------------|----------------------------|--------|
| `resample` | Passed to paradigm; often **1000** for Lee2019 | `get_dataset_sampling_rate` → **1000.0** Hz | Matches raw `fs` in `.mat`. |
| `fmin` / `fmax` | `MotorImagery` defaults **8** / **32** Hz | **8** / **30** Hz | Explicit 8–30 Hz bandpass for Lee2019_MI; BNCI uses `fmax=35` in the same file. |
| Epoch window | If `tmax=None`, full dataset interval **[0, 4]** s | **`tmin=1.0`, `tmax=3.5`** (relative to task) | MOABB `Lee2019` source comments that paper online decoding used **[1, 3.5]** s; this repo adopted that for separability. |
| `events` | `n_classes=2` with `left_hand` / `right_hand` | Same | |
| Channels | EEG from paradigm (typically **62)** | Same **62** × `n_times` in `X` | No EMG in numpy `X` from standard `get_data`. |
| `eval_mode` CrossSession | **LOGO** on `metadata.session` | **LOGO** on `session` column | Same idea as `CrossSessionEvaluation` in MOABB. |
| `eval_mode` WithinSession | MOABB uses **5-fold** per session | Not used in unified runner by default | See `split_comparison.py`. |
| **EEGNet** (`models/eegnet.py`) | N/A (not MOABB) | `ValidSplit(0.2)`, `EarlyStopping`, AdamW `lr=1e-3` | Small-N interaction with 100 trials/session. |
| `max_epochs` | N/A | `globals.get_max_epochs_for_dataset` → **100** for Lee2019_MI (non–CrossSubject) | |
| Underfitting retrain | N/A | `unified_experiment_runner` may retrain without early stopping | See runner when clean ROC-AUC &lt; threshold. |

## Summary

- **Aligned with MOABB** on: sampling rate, bandpass, events, 2-class MI, **CrossSession** session grouping.
- **Deliberate deviation** from “full 4 s” MOABB default: **time crop [1.0, 3.5] s** per MOABB/Lee documentation on the paper’s online window.
- **Training stack** (EEGNet + skorch + `ValidSplit`) is **not** part of MOABB; compare clean metrics to **CSP+LDA** (`linear_baseline_same_splits.py`) to isolate **model vs data**.
