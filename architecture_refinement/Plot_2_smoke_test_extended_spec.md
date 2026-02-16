====================================================================
Plot 2 Smoke Test – Required Fixes and Next Steps (Specification)
====================================================================

Status
------
Plot 2 remains in **SMOKE TEST / DIAGNOSTIC MODE**.
No full-scale (24h) Plot 2 runs are permitted until all criteria below
are satisfied.

This document defines:
1) Required fixes to the current pipeline
2) Minimal next experiments
3) Strict, objective success criteria that gate scale-up

--------------------------------------------------------------------
High-Level Diagnosis (Current State)
--------------------------------------------------------------------

What is NOT broken:
- WS-Flex generator feasibility (Phase 1 PASS at N=20)
- Ability to generate graphs across all regimes (super-sparse → dense)
- Stratified selection logic (Selector C works as intended)

What IS broken or insufficient:
- ORC proxy is strongly correlated with density (Phase 2 FAIL)
- Global TE/ORC selection collapses diversity (expected, now confirmed)
- Smoke test does not yet compute real robustness metrics (AUPC = NaN)
- Integrity diagnostics are incomplete unless legacy mode is allowed
- Sensitivity of robustness metrics to topology has NOT been established

Therefore:
Current Plot 2 results are *not interpretable* and must not be cited.

--------------------------------------------------------------------
REQUIRED FIXES (Must Be Implemented)
--------------------------------------------------------------------

FIX 1 – Smoke Test Must Compute Real Metrics
--------------------------------------------
Problem:
- Phase 5 smoke test launches runners but does not parse outputs.
- results_by_model are NaN; "success" is a placeholder.

Fix:
- Update run_plot2_smoke_test.py to:
  - either write a jobs.csv and invoke analyze_plot2_results, OR
  - directly parse runner CSV outputs and compute:
      * clean ROC-AUC
      * AUPC under AR(1) drift

Required outputs in smoke_test_report.json:
- AUPC per model
- clean ROC-AUC per model
- max_pairwise_delta_AUPC across regimes

Hard fail:
- If any AUPC value is NaN or missing, the smoke test FAILS.

--------------------------------------------------------------------
FIX 2 – Enforce Non-Legacy Integrity Checks
-------------------------------------------
Problem:
- Phase 6 passes only because legacy fingerprint checks are skipped.

Fix:
- For all new smoke and Plot 2 runs:
  - manifest.json MUST exist
  - perturbation fingerprint MUST exist
  - selected_architectures.csv MUST exist

Remove or disable --allow_legacy for smoke tests.

Hard fail:
- Any missing artifact → integrity FAIL → run invalid.

--------------------------------------------------------------------
FIX 3 – Make Proxy Dominance Explicit (No Silent Collapse)
----------------------------------------------------------
Problem:
- ORC_raw correlates ~0.95 with density.
- Pareto width collapses to a single near-dense point.

Fix (choose one path explicitly; do not mix silently):

Path A: Stratified NAS (recommended for Plot 2)
- Treat TE/ORC as *within-regime ranking signals*.
- Selection_method = stratified_by_regime is mandatory.
- Global TE/ORC ranking is forbidden for trained selection.

Path B: Proxy Revision (optional, later)
- Reintroduce clustering and/or path length as objectives or constraints,
  following the original architecture_refinement formulation.
- Re-run Phase 2 to verify Pareto width > 1 and multi-regime coverage.

Hard fail:
- Any trained selection where >50% of models come from one regime,
  unless explicitly justified and logged.

--------------------------------------------------------------------
FIX 4 – Strengthen Phase 2 Diagnostics (Low Cost)
-------------------------------------------------
Problem:
- Phase 2 N=20 is too small to characterize proxy behavior.

Fix:
- Re-run Phase 1 + Phase 2 with N >= 500 (training-free).
- Report:
  - feasibility_rate_by_regime
  - ORC_raw_vs_k correlation
  - Pareto width and regime distribution

Hard fail:
- If Phase 2 diagnostics are not regenerated at larger N,
  proxy conclusions are considered provisional.

--------------------------------------------------------------------
NEXT STEPS (ORDERED, LOW-COMPUTE)
--------------------------------------------------------------------

STEP 1 – Patch Smoke Test Metric Parsing
----------------------------------------
- Implement FIX 1.
- Re-run Phase 5 smoke test (4 graphs, 1 seed, AR(1), SNR=-5).

STEP 2 – Enforce Full Integrity Checks
--------------------------------------
- Implement FIX 2.
- Re-run integrity checks; require full PASS (no legacy bypass).

STEP 3 – Phase 2 at Scale (Training-Free)
-----------------------------------------
- Run Phase 1/2 with N >= 500.
- Confirm whether ORC remains density-dominated.

STEP 4 – Decide NAS Mode Explicitly
-----------------------------------
- Choose Path A (stratified NAS) or Path B (proxy revision).
- Document the choice in the run manifest.

STEP 5 – Minimal Sensitivity Test (Gate)
----------------------------------------
- Using the fixed smoke test:
  - 4 regimes (k≈4,8,14,24)
  - 1 seed
  - AR(1) drift, target_snr_db = -5
- Compute AUPC per model.

--------------------------------------------------------------------
STRICT SUCCESS CRITERIA (GO / NO-GO)
--------------------------------------------------------------------

The Plot 2 Smoke Test is considered **PASSED** only if ALL are true:

1) Smoke test reports numeric AUPC values (no NaNs).
2) Integrity checks PASS without legacy exceptions.
3) max_pairwise_delta_AUPC >= 0.02 across regimes
   (i.e., topology affects robustness at least minimally).
4) Selected models span >= 3 distinct regimes.
5) Perturbation fingerprint confirms AR(1) behavior
   (lag-1 autocorr ≈ rho).

If ANY criterion fails:
- Do NOT run full Plot 2.
- Fix the failing component and repeat the smoke test.

--------------------------------------------------------------------
EXIT CONDITION
--------------------------------------------------------------------

Only after the Plot 2 Smoke Test PASSES may you:
- launch the 24-hour full Plot 2 experiment, and
- interpret NAS vs Random robustness results scientifically.

Until then, all results are diagnostic-only.

====================================================================
End of Specification
====================================================================
