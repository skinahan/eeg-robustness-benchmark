# Lee2019_MI diagnostic toolkit

Scripts support **MOABB-protocol-aligned** checks (CSP + LDA, same `CrossSession` / `WithinSession` ideas as `moabb.evaluations`).

Run from the repo root **or** from this directory (imports use `config` and `_common`):

```bash
cd lee2019_mi_investigation
conda run -n ncp_robustness_proj python data_audit_lee2019_mi.py --subjects 1 2
conda run -n ncp_robustness_proj python baseline_moabb_evaluator.py --subjects 1 2 3 --within-session
conda run -n ncp_robustness_proj python ablation_time_window.py --subjects 1 2
conda run -n ncp_robustness_proj python split_comparison.py --subjects 1 2 3
conda run -n ncp_robustness_proj python linear_baseline_same_splits.py --subjects 1 2
conda run -n ncp_robustness_proj python window_augment_eegnet_experiment.py --subjects 1 2 3 4 --seeds 42 123 456
```

### EEGNet sliding-window augmentation (CrossSession)

`window_augment_eegnet_experiment.py` compares **baseline** (one 1 s crop per outer-train trial) vs **augmented** (1 s windows, 0.5 s stride on outer-train trials only) with EEGNet, `LeaveOneGroupOut` on `metadata["session"]`, **subjects 1–4** by default. Use `--seed 42` for a single run, or **`--seeds 42 123 456`** to repeat the full protocol across multiple experimental seeds (CSV includes a `seed` column). Uses `config.get_paradigm(dataset="Lee2019_MI")` (8–30 Hz bandpass, crop/resample as in `config.py`).

**Leakage checklist**

- Session split first; never slice windows before train/val indices exist.
- Sliding windows only on trials in the outer-training fold; labels copied per window.
- Outer validation: exactly one fixed crop per trial (first 1 s), same for baseline and augmented runs.
- Inner `ValidSplit(0.2)` in `create_eegnet_classifier` sees only outer-train data passed to `fit`.

Outputs: `output/window_augment_eegnet_experiment_summary.csv` and `*_config.json`.

Artifacts are written to `lee2019_mi_investigation/output/` (`*_config.json`, `*_summary.csv`).

See also:

- [`benchmark_reference.md`](benchmark_reference.md) — external citations and comparison rules
- [`config_matrix.md`](config_matrix.md) — MOABB vs this repo
- [`lee2019_mi_investigation_report.md`](lee2019_mi_investigation_report.md) — synthesis
