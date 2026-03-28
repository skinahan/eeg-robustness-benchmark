# Lee2019_MI: external benchmark reference

This note grounds **comparisons** between the repo pipeline and public sources. Always match **metric** (accuracy vs ROC-AUC), **evaluation** (within-session vs cross-session), and **window** (full 4 s vs paper sub-window).

## Lee et al. (2019) — OpenBMI / GigaScience

| Field | Value |
|-------|--------|
| Citation | Lee, M. H., et al. (2019). EEG dataset and OpenBMI toolbox for three BCI paradigms: An investigation into BCI illiteracy. *GigaScience*, 8(5), giz002. |
| DOI | https://doi.org/10.1093/gigascience/giz002 |
| Dataset ID (MOABB) | `Lee2019-MI` (MI paradigm) |

**Takeaways for MI:**

- Recordings at **1000 Hz**, **62 EEG** channels (BrainAmp; nasion reference; AFz ground).
- **Offline** training and **online** test phases are described separately; **BCI illiteracy** (poor per-subject performance) is a **theme** of the paper—so **low CrossSession AUC for some subjects** is not automatically a pipeline bug.
- MOABB’s `Lee2019` MI class documents that the **dataset epoch interval** is `[0, 4]` s for the MI task, while the **paper’s online decoding** used **`[1.0, 3.5]` s** within that window (see `moabb/datasets/Lee2019.py` docstring). This repo’s `get_paradigm(..., dataset="Lee2019_MI")` uses **`tmin=1.0`, `tmax=3.5`** to align with that note.

**What the paper does *not* provide as a single number:** a universal “expected ROC-AUC” for **EEGNet + CrossSession + MOABB preprocessing**—reported figures are paradigm-specific and often **online** accuracy or **offline** curves. Treat **MOABB benchmarks** (below) as the primary numeric comparator for **code** parity.

## MOABB — dataset definition

| Item | Source | Notes |
|------|--------|--------|
| `fs` | Stored in `.mat` as `fs` | Loader sets `Raw` `sfreq` from file; **1000** for released OpenBMI files. |
| MI interval | `Lee2019` → `interval` | Default `[0.0, 4.0]` s in MOABB; paradigm `tmin`/`tmax` crop **within** this. |
| Events | `left_hand`, `right_hand` | Event codes in MOABB; **train** runs are used for classification (`train_run=True`). |
| Scoring `MotorImagery` | `moabb.paradigms.MotorImagery` | `scoring` property → **`roc_auc`** for binary MI. |

## MOABB — evaluation modes (for comparison)

| Mode | Idea | Typical use |
|------|------|-------------|
| **WithinSession** | `StratifiedKFold` (e.g. 5-fold) **within** one session | Easier; estimates “instantaneous” separability. |
| **CrossSession** | `LeaveOneGroupOut` on **session** (`metadata.session`) | **Harder**; train on one session, test on the other (same subject). |

Your `unified_experiment_runner` **CrossSession** matches the **CrossSession** idea (LOGO on sessions). **WithinSession** in MOABB is **not** identical to pooling both sessions—see `split_comparison.py` in this folder.

## MOABB leaderboard / published benchmark numbers

- MOABB maintains **public benchmark results** (e.g. [moabb.github.io](https://moabb.github.io)) and tutorials; exact **ROC-AUC** for EEGNet on Lee2019_MI depends on **MOABB version**, **pipeline** (CSP+LDA vs deep), and **paradigm parameters**.
- **Action:** Re-run the official MOABB tutorial pipeline for your installed `moabb` version, or compare against **`baseline_moabb_evaluator.py`** / **`linear_baseline_same_splits.py`** in this folder, which use the **same paradigm** as [`config.py`](../config.py) and **MOABB-aligned** splits.

## How to compare without apples-to-oranges

1. Prefer **ROC-AUC** (binary) when comparing to `MotorImagery.scoring`.
2. State **WithinSession** vs **CrossSession** explicitly.
3. State **epoch window** (`tmin`, `tmax` or full `[0,4]`).
4. State **subject IDs** (Lee2019 has **high inter-subject variance**).

## References

- Lee et al. (2019), *GigaScience*, DOI 10.1093/gigascience/giz002.
- MOABB: `https://github.com/NeuroTechX/moabb` and `moabb.datasets.Lee2019_MI` docstrings.
