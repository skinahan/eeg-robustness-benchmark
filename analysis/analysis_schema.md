Perfect — this file already contains almost everything we need, and it makes one key design choice easy:

* **Use subject as the inferential unit**
* **Average across seeds** (after filtering to the correct evaluation slice)
* **Compute AUPC from `corrupted_roc_auc` vs `intensity`** (and use `clean_roc_auc` at intensity=0)
* **Use `relative_drop` (or `relative_degradation`) as RD** (with sign convention normalized)

Below is (1) the **step-by-step checklist**, updated to match your actual columns, and (2) a **data-ops spec** formatted as a compact script specification for an LLM coding agent.

---

## 1) Concrete statistical analysis checklist (updated to your schema)

### Step 0 — Decide the analysis slice(s)

1. Pick **metrics** to analyze (primary: `AUPC_roc_auc`, `RD_roc_auc`; sanity: clean vs perturbed `roc_auc`; supporting: `CSV_roc_auc`).
2. Choose which evaluation regimes you will treat as *separate* results sections:

   * `eval_mode ∈ {WithinSession, CrossSession, CrossSubject}` (or whatever exists)
   * `tune ∈ {False, True}` (baseline vs tuned)
3. Filter rows to the actual perturbation evaluation:

   * include only rows where `mode` indicates perturb evaluation (e.g., `test_perturb` and/or `test_perturb_tune`)
   * exclude any training-only artifacts (e.g., sessions or modes that aren’t evaluation)

### Step 1 — Build subject-level ROC-AUC curves (seed-aggregated)

4. For each cell `(dataset, eval_mode, tune, subject, model, noise_type, intensity)` compute:

   * `clean_roc_auc` = mean over seeds (should be constant across intensity in your file; verify)
   * `corrupted_roc_auc` = mean over seeds
   * keep `n_seeds`, and optionally SD over seeds for QA/plots

### Step 2 — Compute robustness summaries per subject

5. For each `(dataset, eval_mode, tune, subject, model, noise_type)` compute:

   * **AUPC** of ROC-AUC vs intensity:

     * Use points `(0, clean_roc_auc)` plus `(intensity, corrupted_roc_auc)` for intensity>0
     * Integrate with trapezoidal rule over intensity
     * Normalize AUPC by dividing by `(max_intensity - 0)` so it’s on the same scale as ROC-AUC
   * **RD** (relative degradation):

     * Use your existing `relative_drop`, but normalize its sign:

       * If `relative_drop = (corrupted - clean)/clean`, then **RD = -relative_drop**
       * If `relative_drop = (clean - corrupted)/clean`, then **RD = relative_drop**
     * Summarize RD across intensities into one per-noise-type value:

       * `RD_mean = mean(RD over intensities>0)`
       * (Optional) `RD_worst = max(RD over intensities>0)` for “worst-case” robustness

### Step 3 — Collapse across noise types for the primary omnibus test

6. For each `(dataset, eval_mode, tune, subject, model)` compute:

   * `AUPC_collapsed = mean_over_noise_types(AUPC)`
   * `RD_collapsed   = mean_over_noise_types(RD_mean)`
   * (Optionally keep noise-type-resolved values for interaction/appendix)

### Step 4 — Primary omnibus tests (per dataset × eval_mode × tune)

7. For each `(dataset, eval_mode, tune)`:

   * Omnibus test on `AUPC_collapsed` across `model` (3-level within-subject):

     * Use RM-ANOVA **if** paired diffs are approximately normal
     * Else Friedman test
   * Repeat omnibus test for `RD_collapsed`

### Step 5 — Planned pairwise contrasts (only if omnibus significant)

8. For each `(dataset, eval_mode, tune, metric)` where omnibus p < α:

   * CNN-NCP vs EEGNet (paired)
   * CNN-NCP vs REEGNet (paired)
   * Test:

     * paired t-test if normal-ish diffs else Wilcoxon signed-rank
   * Multiple comparison correction within family (2 tests) using Holm
   * Report effect size:

     * paired Cohen’s dz + bootstrap 95% CI

### Step 6 — Secondary: noise-type consistency

9. Either:

   * RM-ANOVA with factors `model × noise_type` on AUPC/RD (within-subject), **or**
   * Stratify by `noise_type` and repeat Steps 4–5 (appendix)

### Step 7 — Supporting: clean vs perturbed sanity

10. For each `(dataset, eval_mode, tune, subject, model)` compute a single perturbed summary:

* `corrupted_roc_auc_mean = mean over (noise_type, intensity>0)`

11. Paired test across subjects (within each model):

* compare `clean_roc_auc` vs `corrupted_roc_auc_mean`
* report dz + CI (mostly sanity; keep short)

### Step 8 — Supporting: CSV under perturbation

12. For each `(dataset, eval_mode, tune, model, noise_type, intensity)`:

* `CSV_roc_auc = variance_across_subjects(corrupted_roc_auc_mean_seed)`
* bootstrap CI across subjects (recommended)

13. Compare models descriptively (curves + CI); optionally test AUC of CSV curves.

### Step 9 — Output artifacts

14. Save:

* subject-level analysis table (collapsed + noise-type resolved)
* omnibus results table
* pairwise results table (p_adj, dz, CI)
* CSV table
* dropped/incomplete log

---

## 2) Data operations → compact Python script specification (tailored to your file)

### Script name

`run_stats_paper1.py`

### Purpose

Given a single aggregated CSV like the snippet, produce:

1. a clean subject-level analysis dataset (seed-aggregated),
2. AUPC + RD robustness summaries,
3. omnibus + planned pairwise stats per dataset/eval_mode/tune,
4. CSV curves under perturbation,
5. export CSV + JSON outputs.

---

### Inputs

#### Required CLI args

* `--input_csv PATH` : aggregated results CSV
* `--out_dir PATH`
* `--primary_metric {roc_auc}` (default `roc_auc`; keep extensible)
* `--alpha FLOAT` (default 0.05)
* `--collapse_noise_types {mean,median}` (default mean)
* `--rd_summary {mean,worst}` (default mean)
* `--normalize_aupc {true,false}` (default true)
* `--parametric {auto,true,false}` (default auto)

#### Optional filters (highly recommended)

* `--eval_modes WithinSession CrossSession CrossSubject` (default: all present)
* `--tune_values True False` (default: all present)
* `--models cnn_ncp eegnet reegnet` (default: infer from file)
* `--noise_types eog emg gaussian dropout` (default: all except clean)
* `--mode_regex "test_perturb"` (default: include rows whose `mode` contains `test_perturb`)

---

### Expected columns (from your CSV)

Core identifiers:

* `dataset`, `eval_mode`, `tune`, `subject`, `model`, `noise_type`, `intensity`, `seed`, `mode`

Metric fields used:

* `clean_roc_auc`, `corrupted_roc_auc`, `relative_drop`

(Other columns ignored but preserved optionally.)

---

### Canonicalization rules

* Map `model` to canonical:

  * `cnn_ncp → CNN-NCP`
  * `eegnet → EEGNet`
  * `reegnet → REEGNet`
* Ensure:

  * `subject` is string or int consistently
  * `intensity` float
  * `tune` boolean
* Define “clean” point:

  * Use `clean_roc_auc` as the y-value at intensity=0
  * Treat `noise_type == "clean"` rows as optional; do not require them

---

### Processing pipeline (implement in this order)

#### Step 1 — Load & filter

1. Read CSV into pandas DF.
2. Filter:

   * keep rows where `mode` matches `mode_regex` (default contains `test_perturb`)
   * keep requested `eval_mode` / `tune` / `model` subsets if provided
3. Drop rows with missing `clean_roc_auc` or `corrupted_roc_auc` or `intensity`.

#### Step 2 — Seed aggregation at the curve-point level

4. Group by:

   * `(dataset, eval_mode, tune, subject, model, noise_type, intensity)`
5. Aggregate:

   * `clean_roc_auc_mean = mean(clean_roc_auc over seed)`
   * `corrupted_roc_auc_mean = mean(corrupted_roc_auc over seed)`
   * `relative_drop_mean = mean(relative_drop over seed)`
   * `n_seeds = count`
   * (optional) SDs for QA: `corrupted_roc_auc_sd`, etc.
6. Output DF: `df_points`

#### Step 3 — Completeness checks & logging

7. For each `(dataset, eval_mode, tune, subject, noise_type)` verify all 3 models exist across intensities.
8. For each `(dataset, eval_mode, tune, subject, model, noise_type)` verify there are ≥2 intensity points (>0) for AUPC; else mark incomplete.
9. Write `dropped_units.log` listing removed units and reasons.
10. Filter out incomplete units *only for the analyses that require them* (do not globally drop unless necessary).

#### Step 4 — Compute AUPC per subject/model/noise_type

11. For each `(dataset, eval_mode, tune, subject, model, noise_type)`:

* Build arrays:

  * x = [0.0] + sorted(unique intensities>0)
  * y = [clean_roc_auc_mean_at_any_intensity] + corrupted_roc_auc_mean_at_each_intensity

    * (clean_roc_auc_mean is typically constant across intensities; use mean across intensities to be safe)
* Compute `aupc = trapz(y, x)`
* If `normalize_aupc`: `aupc /= (max(x) - min(x))`

12. Output DF: `df_aupc` with columns:

* `dataset, eval_mode, tune, subject, model, noise_type, aupc_roc_auc`

#### Step 5 — Compute RD per subject/model/noise_type

13. Define sign convention:

* For each group, compute an empirical check:

  * if median(relative_drop_mean) < 0 (typical when corrupted>clean yields negative drop in your snippet), set `RD = -relative_drop_mean`
  * else `RD = relative_drop_mean`
* (Also allow override flag `--rd_sign {auto,negate,identity}`)

14. Summarize RD over intensities>0 per group:

* `rd_mean = mean(RD over intensity>0)`
* `rd_worst = max(RD over intensity>0)`

15. Output DF: `df_rd` with:

* `dataset, eval_mode, tune, subject, model, noise_type, rd_mean, rd_worst`

#### Step 6 — Build inference dataset (collapsed + resolved)

16. Merge `df_aupc` and `df_rd` on keys.
17. Create collapsed metrics over noise types for each `(dataset, eval_mode, tune, subject, model)`:

* `aupc_collapsed = mean/median over noise_type`
* `rd_collapsed = mean/median over noise_type` using chosen summary (`rd_mean` or `rd_worst`)

18. Save:

* `analysis_subject_level_resolved.csv` (noise-type resolved)
* `analysis_subject_level_collapsed.csv` (collapsed)

#### Step 7 — Statistical testing helpers

19. For each `(dataset, eval_mode, tune)` and for each metric in `{aupc_collapsed, rd_collapsed}`:

* Pivot to wide: index `subject`, columns `model`, values metric
* Drop subjects missing any model (paired requirement)

20. Assumption diagnostics (if `parametric=auto`):

* Compute paired diffs:

  * `d1 = CNN-NCP - EEGNet`
  * `d2 = CNN-NCP - REEGNet`
* Shapiro-Wilk on `d1` and `d2`
* If both p > 0.05 → treat as parametric; else nonparametric

#### Step 8 — Omnibus tests

21. If parametric:

* Run repeated-measures ANOVA with within factor `model`
* Record F, p, partial_eta2

22. Else:

* Run Friedman test across the three model columns
* Record statistic, p, Kendall_W

Write `stats_omnibus.csv`.

#### Step 9 — Planned pairwise tests (conditional)

23. If omnibus p < alpha:

* CNN-NCP vs EEGNet:

  * paired t-test or Wilcoxon
  * compute Cohen’s dz
  * bootstrap CI for dz (resample subjects, e.g. 10,000 reps)
* CNN-NCP vs REEGNet: same

24. Apply Holm correction to the two p-values within the family `(dataset, eval_mode, tune, metric)`.
25. Write `stats_pairwise.csv`.

#### Step 10 — CSV under perturbation (supporting)

26. From `df_points` (seed-mean already):

* For each `(dataset, eval_mode, tune, model, noise_type, intensity)` compute:

  * `csv = variance across subjects of corrupted_roc_auc_mean`
  * (optional) bootstrap CI for variance by resampling subjects

27. Save `csv_by_level.csv`.

#### Step 11 — JSON summary

28. Write `stats_summary.json` with nested keys:

* dataset → eval_mode → tune → metric → {omnibus, pairwise}

---

### Outputs (exact filenames)

* `analysis_subject_level_resolved.csv`
* `analysis_subject_level_collapsed.csv`
* `stats_omnibus.csv`
* `stats_pairwise.csv`
* `csv_by_level.csv`
* `stats_summary.json`
* `dropped_units.log`

---

### Libraries

* `pandas`, `numpy`
* `scipy.stats`
* `statsmodels` **or** `pingouin` (RM-ANOVA)
* `statsmodels.stats.multitest` (Holm)
* `json`, `pathlib`, `argparse` (or config dict)

---

### Notes (important)

*  `relative_drop` can be **negative** (because corrupted > clean at some intensities). That’s fine, but for RD you likely want “degradation” to be **positive when performance drops**. The script should enforce a consistent sign convention (Step 5).
* `clean_roc_auc` repeats across intensities; we should take the **mean clean** within group to be safe. It's very important these groups are well-defined (i.e. not split across subjects)
*  
