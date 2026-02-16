====================================================================
Paper 3 / Plot 2 – Locked Experiment Specification (v1.0)
====================================================================

Document purpose
----------------
This specification formally locks:
(A) the intermediate “mini-scale” Plot 2 run design (fast validation),
(B) the final full-scale Plot 2 run design (publishable evidence),
(C) required code changes (new mechanisms vs. parameter updates),
(D) required outputs and strict success criteria.

Scope note
----------
Plot 2 evaluates whether training-free topology optimization (TPE over
graph-space proxies) yields architectures with improved robustness
relative to (i) random WS-Flex graphs and (ii) an external random wiring
baseline. Robustness is evaluated under correlated perturbations
(AR(1) drift primary), in cross-session MI decoding.

This spec assumes:
- Model family: CNN–CfC with masked-weight wiring (fixed sparsity mask)
- Hidden size: H = 32 (locked)
- Dataset: BNCI2014_001 MI (cross-session)

====================================================================
1) Research Questions and Claims (Plot 2)
====================================================================

RQ2.1: Does topology (wiring sparsity structure) measurably affect
      robustness under temporally correlated perturbations?

RQ2.2: Does training-free multi-objective search (TPE over TE/ORC proxies)
      produce topologies that improve robustness relative to:
      - Random selection from the same WS-Flex family (controlled baseline)
      - External random wiring class (out-of-family baseline)?

RQ2.3: Which robustness summaries are sensitive to topology effects?
      (AUPC vs worst-case drop under maximum perturbation.)

Locked claim targets:
- Plot 2 will report robustness using BOTH:
  (Primary) worst-case degradation (max_drop)
  (Secondary) AUPC over intensity
- Plot 2 comparisons are made under stratified regime coverage to avoid
  collapse to near-dense graphs.

====================================================================
2) Locked Core Design Choices
====================================================================

2.1 Dataset / protocol (locked)
- Dataset: BNCI2014_001 (Motor Imagery)
- Evaluation: cross-session only

2.2 Model testbed (locked)
- Model: CNN–CfC (masked-weight wiring via fixed sparsity mask)
- Hidden size: H = 32
- All non-wiring hyperparameters: fixed across models within a run.

2.3 Perturbations (locked)
Primary perturbation (main Plot 2):
- AR(1) temporally correlated drift
- rho = 0.97
- target_snr_db = −5
- intensity grid: alpha ∈ {0, 0.25, 0.5, 0.75, 1.0} * alpha_max

Optional perturbations (appendix-ready; not required for Plot 2 acceptance):
- Spatially correlated Gaussian (ell = median electrode distance)
- EMG band-limited noise (20–80 Hz, envelope off by default)
These may be evaluated after Plot 2 primary is complete.

2.4 Metrics (locked)
For each model:
- clean_roc_auc (alpha = 0)
- robustness curve roc_auc(alpha)
- AUPC = mean_{grid} roc_auc(alpha) with fixed grid above
- max_drop = roc_auc(0) − roc_auc(alpha_max)
- mid_drop = roc_auc(0) − roc_auc(alpha=0.5*alpha_max)

Primary robustness metric for statistical comparisons:
- max_drop (lower is better)

Secondary robustness metric:
- AUPC (higher is better)

2.5 Architecture sets (locked)
We compare four groups:

G0) Baseline (fixed):
- “baseline” model wiring: NCP default wiring (existing baseline)
  OR your established baseline wiring for CNN–CfC (choose ONE and keep fixed)

G1) Random WS-Flex (controlled):
- Randomly generated WS-Flex graphs, selected via stratified regime selection

G2) TPE WS-Flex (proposed):
- Graphs produced by Optuna TPE multi-objective search over TE/ORC proxies,
  selected via stratified regime selection

G3) External Random Wiring (out-of-family control):
- ncps package “random wiring” class (or equivalent) used to generate
  wiring masks not constrained to WS-Flex
- Selected with matched mask-density stratification (see §6.3)

====================================================================
3) Graph Space and Regime Stratification
====================================================================

3.1 WS-Flex generator (locked family)
- Use the existing WS-Flex generator family.

3.2 Search parameters (locked bounds for Plot 2 v1.0)
- k_degree: integer in [2, 26]
- p_rewiring: float in [0.0, 1.0]
- Any additional WS-Flex constraints are allowed, but must be logged:
  (target_clustering, target_path_length if used)

3.3 Regime bins (locked)
Define regimes by k bins:
- super_sparse: k ∈ [2, 6]
- sparse:       k ∈ [7, 12]
- moderate:     k ∈ [13, 18]
- near_dense:   k ∈ [19, 26]

3.4 Stratified selection (locked mechanism)
All trained sets (Random WS-Flex and TPE WS-Flex) MUST be selected with
regime stratification to avoid collapse:

Given total B graphs to train per method:
- allocate B_r = B / 4 per regime (integer; distribute remainder round-robin)
- within each regime:
  - rank candidates by Pareto rank (TE/ORC) then crowding distance,
    OR (if not available) by within-regime scalar score:
      score = z(TE_raw within regime) + z(ORC_raw within regime)
- if a regime has insufficient candidates:
  - fill from the nearest regime(s) while minimizing collapse_score
- report “collapse_score” = max_regime_fraction; must be ≤ 0.50.

====================================================================
4) Intermediate Step (“Mini-Scale Plot 2”) – Design
====================================================================

Purpose
- Validate that Plot 2 effects persist beyond a single subject/seed and
  that selection + pipelines produce interpretable results before a full run.

Design (locked)
- Subjects: S_small = 3 subjects (recommended default; minimum 2)
- Seeds: 1 seed
- Architectures per group:
  - baseline: 1
  - random_ws_flex: B_small = 8 (2 per regime)
  - tpe_ws_flex:    B_small = 8 (2 per regime)
  - external_random: B_small_ext = 8 (density-matched; see §6.3)

Total models (default): 1 + 8 + 8 + 8 = 25 models
This should be dramatically faster than the full experiment while still
testing generality.

Mini-scale success criteria (strict)
- Integrity criteria all pass (§7)
- Stratification criteria met:
  - collapse_score ≤ 0.50 for random_ws_flex and tpe_ws_flex
- Topology sensitivity persists:
  - bootstrap 95% CI for (tpe_ws_flex − random_ws_flex) in max_drop is
    not absurdly wide (operational threshold: CI width ≤ 0.10)
  - directionally consistent effect across ≥2 of 3 subjects
    (sign agreement on tpe − random in max_drop)

Decision after mini-scale:
- If mini-scale passes: proceed to full-scale.
- If mini-scale fails: stop and diagnose (selection, perturbation, or metric).

====================================================================
5) Full Plot 2 – Final Design
====================================================================

Subjects and seeds (locked target; adjust only if compute forces)
- Subjects: all available in BNCI2014_001 cross-session protocol
- Seeds: 3 seeds (recommended default; minimum 2)

Architectures per group (locked)
- baseline: 1
- random_ws_flex: B = 12 (3 per regime)
- tpe_ws_flex:    B = 12 (3 per regime)
- external_random: B_ext = 12 (density-matched; see §6.3)

Total models (default): 1 + 12 + 12 + 12 = 37 models per seed
Total training jobs: 37 * n_seeds

Full-scale success criteria (strict, publishable)
- Integrity criteria all pass (§7)
- Stratification criteria met:
  - collapse_score ≤ 0.50 for each method’s selected set
- Statistical claim tests (primary):
  - Compare tpe_ws_flex vs random_ws_flex on max_drop:
    * report mean difference, bootstrap 95% CI
    * report effect size (Cohen’s d or rank-biserial)
  - Compare tpe_ws_flex vs external_random on max_drop:
    same reporting
- Secondary metrics:
  - AUPC comparisons with same reporting (not required to “win”)

Interpretation constraint:
- If max_drop improves but AUPC does not, still acceptable:
  Plot 2 emphasizes worst-case robustness, not only average-case.

====================================================================
6) Required Mechanisms and Implementation Changes
====================================================================

6.1 Existing functionality (parameter-only updates)
- WS-Flex generator (already implemented)
- Optuna TPE multi-objective study (already implemented)
- Evaluation pipeline (cross-session, MI) (already implemented)
- AR(1) perturbation injection (already implemented)

Parameter-only changes required:
- Ensure H = 32
- Ensure AR(1): rho=0.97, target_snr_db=-5
- Ensure fixed alpha grid and alpha_max calibration

6.2 Required updates to existing scripts (must implement)
UPDATE A: Selection policy for trained graphs
- Replace any global top-B selection with stratified selection (§3.4)
- Ensure it applies identically to:
  - random_ws_flex candidate pool
  - tpe_ws_flex candidate pool

UPDATE B: Analysis outputs must include max_drop and mid_drop
- Ensure analyze_plot2_results (or equivalent) computes and exports:
  - AUPC
  - max_drop
  - mid_drop
- Update plotting scripts to show:
  - Primary: max_drop comparisons (box/violin + CI)
  - Secondary: AUPC comparisons

UPDATE C: Mini-scale runner configuration
- Add a config mode that runs:
  - S_small subjects
  - 1 seed
  - B_small selection per group

6.3 New mechanisms to implement (small, required)
NEW 1: External random wiring baseline (out-of-family)
- Implement a generator wrapper that produces wiring masks using the
  ncps random wiring class (or equivalent).
- Add density-matched stratified selection for these masks:
  - Compute mask_density for each candidate mask
  - Define density bins aligned to WS-Flex regime densities (or quantiles)
  - Select B_ext masks matching the WS-Flex density distribution
- Log selection summaries.

NEW 2: Full integrity/fingerprint enforcement
- Ensure perturbation_fingerprint.json is always produced for runs
  (no legacy bypass allowed for Plot 2).
- Fingerprint must include:
  - lag1_autocorrelation (target ~0.97)
  - perturbation_type
  - target_snr_db and empirical_snr_db

NEW 3: Run manifest schema enforcement
- Every run writes manifest.json containing:
  - perturbation type and params
  - selection_method
  - generator bounds
  - TE/ORC formulas used
  - selected_architectures.csv path
  - code version id + seeds + dataset info

====================================================================
7) Integrity and Logging Requirements (Hard Gates)
====================================================================

For EVERY run (mini and full), the following MUST exist:

Artifacts:
- manifest.json
- selected_architectures.csv
- perturbation_fingerprint.json
- per_seed_metrics.csv (must include max_drop and AUPC)
- per_graph_metrics.csv (aggregated)

Hard fail conditions:
- fingerprint missing OR lag1_autocorrelation < 0.90 for AR(1)
- missing max_drop in outputs
- duplicate model identifiers in selected set
- collapse_score > 0.50 for any group’s selected set
- mismatch between plot labels and manifest perturbation_type

====================================================================
8) Reporting Deliverables (What Plot 2 Must Produce)
====================================================================

Mini-scale deliverables:
- A single figure or table:
  - max_drop distribution for baseline, random_ws_flex, tpe_ws_flex,
    external_random (3 subjects aggregated)
- Short textual summary:
  - whether tpe improves max_drop vs random, directionally consistent

Full-scale deliverables:
- Main Plot 2 figure:
  - max_drop: baseline vs random_ws_flex vs tpe_ws_flex vs external_random
  - include bootstrap 95% CI for key deltas (tpe-random, tpe-external)
- Secondary figure or appendix:
  - AUPC comparisons
- Selection summary appendix:
  - regime counts, k distribution, mask density summary per group

====================================================================
9) Recommended Defaults (If Optional Choices Arise)
====================================================================

- If forced to choose one robustness metric for headline claim:
  choose max_drop.
- If TE/ORC proxy selection conflicts with diversity:
  prioritize stratified selection; treat TE/ORC as within-regime ranking.
- If compute is limited:
  reduce number of seeds before reducing subjects.
- Keep rho fixed at 0.97; adjust only if fingerprint indicates mismatch.

====================================================================
End of Specification
====================================================================
