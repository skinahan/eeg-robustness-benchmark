====================================================================
Plot 2 Smoke Test – Concrete Implementation Spec (Revised)
====================================================================

Purpose
-------
This smoke experiment determines whether robustness metrics are
sensitive to large differences in recurrent topology under the current
CNN–CfC (masked-weight) implementation.

This spec enumerates:
- exact experimental changes,
- which are parameter-only vs. code changes,
- and strict success criteria gating scale-up.

--------------------------------------------------------------------
Big Questions
--------------------------------------------------------------------

Q1) Do *extreme* WS-Flex topologies (very sparse → very dense) produce
    measurable robustness differences?

Q2) Is AUPC under AR(1) drift a sufficiently sensitive robustness metric
    for topology in this model family?

Q3) Are topology effects being masked by implementation details
    (CNN dominance, dense self-connections, masked-weight wiring)?

--------------------------------------------------------------------
Experiment Definition (What Will Be Run)
--------------------------------------------------------------------

Models (4 total, fixed, no search):
- super_sparse: k = 2
- sparse:       k = 6
- moderate:     k = 14
- near_dense:   k = 26

Common configuration:
- Model: CNN–CfC (masked-weight wiring)
- Hidden size: H = 32
- Training: 1 seed, 1 subject, smoke-length training
- Evaluation: cross-session
- Perturbation: AR(1) drift

Perturbation defaults:
- rho = 0.97
- target_snr_db = −5
- alpha grid = {0, 0.25, 0.5, 0.75, 1.0} · alpha_max

--------------------------------------------------------------------
Metrics to Compute
--------------------------------------------------------------------

Already supported:
- clean ROC-AUC
- AUPC (if analyze_plot2_results is used)

New / additional metrics:
- max_drop = ROC-AUC(0) − ROC-AUC(alpha=1)
- mid_drop = ROC-AUC(0) − ROC-AUC(alpha=0.5)

--------------------------------------------------------------------
Implementation Breakdown
--------------------------------------------------------------------

(A) PARAMETER-ONLY CHANGES (No new code)
---------------------------------------
These require *only config edits or CLI arguments*.

1) Topology contrast
   - Explicitly set k = {2, 6, 14, 26} in the smoke config.
   - Disable TPE and random sampling entirely.

2) Perturbation severity
   - target_snr_db = −5 (already implemented).
   - Optionally retry with −10 if needed (Decision Point A).

3) Selection
   - selection_method = fixed (no selector logic invoked).

--------------------------------------------------------------------
(B) UPDATES TO EXISTING SCRIPTS (Small Edits)
---------------------------------------------

B1) Smoke test metric parsing  ❗ REQUIRED
   Script: run_plot2_smoke_test.py

   Current state:
   - Launches runners
   - Does not parse runner outputs
   - Writes NaNs and placeholder success

   Required update:
   - Reuse existing analyze_plot2_results logic OR
   - Inline minimal parsing of runner CSVs to compute:
       * AUPC
       * clean ROC-AUC
       * max_drop
       * mid_drop

   Output:
   - smoke_test_report.json must contain numeric values.

B2) Mask statistics logging  ❗ REQUIRED
   Script: unified_experiment_runner or model init path

   Add logging for:
   - total possible recurrent edges
   - active mask edges
   - mask density = active / possible
   - per-layer mask sizes (if multi-layer)

   These values must be written to:
   - per-model JSON
   - smoke_test_report.json

   NOTE:
   This is a read-only inspection of existing masks, not a model change.

B3) Perturbation fingerprint enforcement
   Script: integrity / diagnostics utilities

   Ensure that:
   - lag-1 autocorrelation of injected noise is computed
   - fingerprint is stored and checked for smoke runs

--------------------------------------------------------------------
(C) NEW SMALL HELPER FUNCTIONS (Targeted, Minimal)
--------------------------------------------------

C1) Robustness summary helper (recommended)
   New helper:
     compute_smoke_robustness_metrics(results_df)

   Responsibilities:
   - compute AUPC
   - compute max_drop
   - compute mid_drop
   - return dict for JSON serialization

   Rationale:
   - Avoid duplicating analysis logic
   - Keep smoke script short and robust

C2) Mask inspection utility
   New helper:
     summarize_wiring_mask(model) → dict

   Returns:
   - n_active_edges
   - n_possible_edges
   - mask_density
   - (optional) per-layer stats

   Rationale:
   - Centralize mask inspection
   - Reuse in later ablations if needed

--------------------------------------------------------------------
Success / Failure Criteria (Hard Gates)
--------------------------------------------------------------------

Primary gate (required):
- max_pairwise_delta ≥ 0.02 for *any* robustness metric
  (AUPC OR max_drop OR mid_drop)

Secondary gates:
- mask_density differs across regimes by ≥2×
- perturbation fingerprint confirms AR(1) (lag-1 ≈ rho)

If ALL pass:
- Topology sensitivity established.
- Proceed to scaled Plot 2 with stratified selection.

If PRIMARY gate fails:
- Do NOT scale Plot 2.
- Proceed to Decision Points below.

--------------------------------------------------------------------
Decision Points (Only If Primary Gate Fails)
--------------------------------------------------------------------

Decision A – Increase perturbation severity (parameter-only)
- Set target_snr_db = −10
- Re-run the same 4 models.

Decision B – Reduce CNN dominance (parameter change)
- Reduce CNN channels or layers in smoke config only.
- Re-run the same 4 models.

Decision C – Increase recurrent influence (parameter change)
- Increase H to 64 OR
- Reduce temporal pooling / increase sequence length.

Only one decision is tested at a time.

--------------------------------------------------------------------
Constraints
--------------------------------------------------------------------

- No full Plot 2 runs
- No NAS / TPE
- No changing model class
- No mixing perturbations
- No additional seeds or subjects

--------------------------------------------------------------------
Outcome
--------------------------------------------------------------------

This smoke experiment conclusively determines whether Plot 2 can reveal
topology-dependent robustness effects under the current modeling and
metric choices, or whether architectural/metric revisions are required
before scaling.

====================================================================
End of Spec
====================================================================
