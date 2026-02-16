================================================================================
PLOT 2 — MINI-SCALE RUN (LOCKED NEXT STEPS SPEC)
Version: vMiniLock_2026-02-16
Scope: Execute ONE decisive mini-scale Plot 2 run under a topology-sensitive
       perturbation regime, with locked configuration, clear go/no-go criteria,
       and full reporting (clean + robustness + stability metrics).
================================================================================

0) CONTEXT / WHAT WE JUST LEARNED (INTERMEDIATE DIAGNOSTIC)
-----------------------------------------------------------
We executed:
  run_plot2_intermediate_diagnostic --generator_mode ws_flex --target_snr_db -12

Observed outcomes (must be cited in Plot 2 run notes / manifest):
  - max_pairwise_delta_max_drop = 0.2758  (threshold was 0.03; huge separation)
  - effect_size_pass = True
  - regime separation achieved in 3/4 regimes (super_sparse, sparse, near_dense)
  - moderate regime was near-tied (not a failure; indicates non-uniformity)

Interpretation:
  - Topology effects are real and can be large, BUT only become visible under
    sufficiently strong correlated / nonstationary perturbation.
  - Earlier weak/no-effect Plot 2 outcomes were likely dominated by an
    under-stressing perturbation regime, not necessarily a failed proxy space.
  - Small-world vs non-small-world directionality can be regime-dependent;
    our goal is not to pre-assume SW always wins, but to reliably measure
    topology-conditioned robustness under fair capacity control.

Motivation for next steps:
  - Now that we have a perturbation configuration that exposes topology
    sensitivity, we proceed to a mini-scale A/B/C run to validate the full
    “inversion of Waqas” structure with minimal training cost, and only then
    scale up.

Non-goal:
  - Do NOT introduce modular_ws_flex into Plot 2 mainline at this stage.
    It failed proxy-viability comparisons and currently reduces manifold
    coverage/diversity. Keep modular as a separate/appendix branch only.

================================================================================
1) MINI-SCALE GOAL (WHAT WE MUST ESTABLISH)
-------------------------------------------
We must establish, in a single mini-scale run, that:
  (G1) A/B/C baseline structure is implemented correctly and non-degenerate.
  (G2) The chosen proxy space produces measurable differences under training:
       A vs B demonstrates proxy validity (static selection helps).
       B vs C demonstrates adaptive benefit (TPE helps beyond static filtering).
  (G3) Results are not confounded by low clean performance:
       Robustness comparisons must be reported alongside clean ROC-AUC and
       include robustness metrics that correct for baseline performance.

================================================================================
2) LOCKED EXPERIMENT CONFIGURATION (DO NOT CHANGE)
--------------------------------------------------

2.1 Dataset / split / subjects
- Dataset: BNCI2014_001 (MI)
- Evaluation mode: Cross-Session ONLY
- Subjects: [1, 3, 4]  (explicitly skip subject 2 due to known issues)
- Training protocol: identical across all models; same preprocessing as prior
  Plot 2 runs (no new preprocessing introduced here).

2.2 Base model (capacity control)
- Model family: CNNWiredCfCMin (or the locked Plot 2 minimal CNN + CfC chamber)
- Hidden dimension: H = 32 (locked)
- CfC cell hyperparameters: locked to current Plot 2 defaults
- Feature extractor (CNN) hyperparameters: locked to current Plot 2 defaults
- Edge parameterization: masked-weight wiring (mask multiplies dense weights)
- Orientation policy for hidden adjacency: locked deterministic policy
  (must be identical across all WS-Flex graphs and across seeds).

Rationale:
- We must keep capacity as constant as our framework allows and attribute
  differences to topology, not hidden size, not feature extractor differences.

2.3 Perturbation (primary stress test)
- Primary perturbation type: ar1_drift
- Target SNR: target_snr_db = -12   (LOCKED; validated by intermediate test)
- Intensity grid: fixed 5-point alpha grid from 0 to 1 inclusive:
    alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
  where alpha=1 corresponds to the maximum perturbation under target_snr_db=-12.
- Fingerprint requirement: perturbation_fingerprint.json must exist and report
  lag1_autocorrelation consistent with AR(1) (non-trivial, e.g., > 0.9 typical
  in previous successful fingerprints).

Optional (NOT in this mini run unless already free):
- Additional perturbation types (eog, dropout, gaussian) are NOT required for
  this mini-scale gate. Keep scope tight. Add later only after mini passes.

2.4 Training seeds / budgets
- Seeds per topology: S = 1 (mini-scale speed)
- Training seed list: one deterministic seed (recorded in manifest)
- Number of selected graphs per method:
    B = 12 total per WS-Flex method group CANDIDATES, but selection must be
    coverage-aware and regime-stratified:
      - For A: select 8 graphs total (2 per regime) OR 12 graphs total (3 per
        regime) depending on current Plot 2 mini defaults. Pick ONE and lock:
        RECOMMENDED: B_mini = 8 (2 per regime) for faster completion.
      - For B: same count as A, selected from the SAME pool as A.
      - For C (TPE): same count as A/B.

Locked recommendation:
- Use B_mini = 8 graphs per WS-Flex method (A, B, C), i.e., 24 WS-Flex graphs
  total, + NCP baseline as a single additional model (optional), with:
    - A: 8 graphs (true random within bins)
    - B: 8 graphs (static proxy-selected from A’s pool)
    - C: 8 graphs (TPE adaptive, then selected with same bin quotas)

2.5 Baseline structure (reviewer-proof “inversion”)
A = True Random WS-Flex baseline
- Sample uniformly from the WS-Flex constrained space (connectedness + any
  capacity constraints that are applied equally to all methods).
- Selection is RANDOM within fixed bins (no proxy ranking).

B = Static Proxy WS-Flex baseline
- Uses the IDENTICAL candidate pool as A.
- Selection uses the proxy (Pareto/crowding in proxy space) within bins.

C = Adaptive Proxy NAS (Optuna TPE)
- Multi-objective TPE search in the SAME proxy space; produces its own pool.
- Then selected using the SAME bin-aware quotas/strategy as A/B.

Proxy space (locked for this mini run):
- Primary objectives: (TE, sigma_small_world)
- Tie-breaker only: ORC_res (residualized within k), and optionally ρ_norm if
  it is already implemented as a constraint (do not add new constraints here).

Binning / diversity:
- coverage_level = "regime_cl_bins_fixed"
- Bin edges for C and L are frozen in manifest from the latest proxy viability
  run that passed (including any relaxed gates used; these must be logged).

================================================================================
3) OUTPUT METRICS TO REPORT (MINI RUN MUST PRODUCE ALL)
-------------------------------------------------------

3.1 Clean performance (required)
- clean ROC-AUC per (subject, seed, graph)
- aggregate clean ROC-AUC per graph and per method (A/B/C)

3.2 Robustness metrics (match Paper 1/2 reporting)
For primary perturbation ar1_drift:
- AUPC(alpha): normalized area under perturbation curve
- RD(alpha): relative degradation at each alpha
  Define for each seed/subject/graph:
    Let m0 = metric(alpha=0)  (clean ROC-AUC)
    Let m(alpha) = ROC-AUC at alpha
    RD(alpha) = (m0 - m(alpha)) / max(eps, m0)
- max RD: max over alpha grid (excluding alpha=0 unless your implementation includes it)
    maxRD = max_{alpha in grid} RD(alpha)

Also keep:
- max_drop: m0 - m(alpha=1)
  NOTE: treat max_drop as supporting only; interpret alongside clean ROC-AUC and RD.

3.3 Stability / generalization spread (CSV)
We must report cross-subject variance to avoid “robust but inconsistent” graphs.
Define CSV for each graph:
- Compute subject-level mean clean ROC-AUC for each subject s:
    μ_s = mean over seeds of clean ROC-AUC for subject s
- CSV_clean = Var_s(μ_s)   (or Std_s if that is what Paper 1 used; be consistent)
Similarly for robustness:
- CSV_AUPC = Var_s( mean over seeds of AUPC(alpha) for subject s )
- CSV_maxRD = Var_s( mean over seeds of maxRD for subject s )

If Paper 1/2 defined CSV differently (e.g., std of subject means), use that
definition exactly and note it in the report header.

Required deliverables:
- per_seed_metrics.csv: includes clean ROC-AUC, AUPC, RD curve values, maxRD,
  max_drop, mid_drop, plus topology metadata (k, p, C, L, sigma, TE, ORC_res)
- per_graph_metrics.csv: aggregated mean/std across seeds and subjects
- bootstrap_diff.json: hierarchical bootstrap comparisons (see Section 5)
- report.txt: includes gates + summary tables + sanity checks

================================================================================
4) MINI-SCALE EXECUTION ORDER (STRICT, WITH GO/NO-GO)
-----------------------------------------------------

Stage 0 — Preflight integrity (GO/NO-GO)
- Confirm perturbation fingerprint exists and matches expected AR(1) behavior.
- Confirm selection pools produce no duplicate model_name and no reused graph_id.
GO criteria:
  - fingerprint exists
  - duplicates = 0

Stage 1 — Candidate generation + selection (A/B/C) (GO/NO-GO)
- Generate candidate pool for A/B (shared pool) and candidate pool for C (TPE).
- Apply fixed (regime, Cbin, Lbin) selection.
GO criteria:
  - A/B/C each select exactly B_mini graphs
  - A selection is uniform random within bins (no proxy usage)
  - B selection uses proxy ranking within bins and differs materially from A
  - Overlap constraints:
      overlap(A,B) is allowed to be high because same pool, but A and B MUST
      not be identical (e.g., overlap(A,B) <= 75% OR at least 2 graphs differ)
      overlap(B,C) <= 50% (preferred) or must pass your existing overlap gate.
  - Selected sets span ≥2 regimes and ≥5 occupied (C,L) cells (mini feasibility)

If Stage 1 fails: STOP. Fix selection logic before training.

Stage 2 — Training + evaluation (mini) (GO/NO-GO)
- Train/evaluate all jobs for subjects [1,3,4], seed S=1, perturbation ar1_drift,
  alpha_grid locked.
GO criteria:
  - 100% job completion
  - No identical-result pathology (e.g., all ROC-AUC identical across graphs)
  - Clean ROC-AUC non-trivial (e.g., not all near chance)

Stage 3 — Analysis + decision (GO/NO-GO)
Compute hierarchical bootstrap (graph-resample then seed-resample):
Primary comparisons (robustness):
- A vs B (proxy validity):
    Prefer lower maxRD and/or higher AUPC for B vs A, while maintaining clean ROC-AUC.
- B vs C (adaptive benefit):
    Prefer C better than B under same proxy.

We must avoid “robust because bad.”
Therefore define a clean-performance floor:
- For each graph, require clean ROC-AUC >= (median_clean_over_all_graphs - 0.02)
  OR mark the graph as “low-clean” and exclude from the primary robustness claim.

Primary GO criteria (mini-scale):
At least ONE of the following must be true, with consistent direction across
>=2 of 3 subjects:
  (P1) mean(maxRD_B - maxRD_A) < 0 and bootstrap 95% CI excludes 0
  (P2) mean(AUPC_B - AUPC_A) > 0 and bootstrap 95% CI excludes 0
AND
  (P3) mean(cleanROC_B - cleanROC_A) >= -0.01 (no meaningful clean drop)

Adaptive GO criteria:
At least ONE of:
  (A1) mean(maxRD_C - maxRD_B) < 0 with CI excluding 0
  (A2) mean(AUPC_C - AUPC_B) > 0 with CI excluding 0
AND
  (A3) mean(cleanROC_C - cleanROC_B) >= -0.01

Supporting (not required, but reported):
- Effect sizes (Cohen’s d) for (B-A) and (C-B) on maxRD and AUPC.
- CSV metrics: show whether improvements come with increased variability.

If Stage 3 fails:
- Do NOT scale up.
- Revisit: proxy viability (bin edges), overlap collapse, and/or perturbation
  severity confirmation. However, do NOT reduce perturbation severity below
  -12 dB, since intermediate diagnostic indicates topology sensitivity there.

================================================================================
5) EXPECTED OUTCOMES (WHAT WE SHOULD SEE IF THINGS ARE WORKING)
--------------------------------------------------------------
Given intermediate results at target_snr_db=-12, we expect:
- Non-trivial spread in robustness across graphs (already confirmed in principle).
- Clean ROC-AUC will vary somewhat with topology, but should not collapse.
- A vs B should show some improvement if TE+σ proxy has predictive value.
- B vs C may be smaller; if it is null, that still supports a useful narrative:
  “proxy filtering works; adaptivity yields limited extra benefit at this budget.”

We do NOT require “small-world always wins.”
We require “proxy space + selection is non-degenerate and measurable under stress.”

================================================================================
6) IMPLEMENTATION NOTES / REQUIRED PATCHES (IF ANY)
---------------------------------------------------
This mini run assumes the following are already present; if missing, implement
as PATCHES BEFORE Stage 2 training:

- Existing functionality (verify):
  [ ] run_plot2_topology_study supports:
      - subjects override to [1,3,4]
      - ar1_drift with target_snr_db = -12 and alpha_grid
      - A/B/C baseline structure and bin-fixed selection
  [ ] analyze_plot2_results computes:
      - clean ROC-AUC per seed/subject/graph
      - AUPC(alpha)
      - RD(alpha) and maxRD
      - CSV metrics (per Paper 1/2 definition)
      - hierarchical bootstrap at graph->seed nesting

- New additions (implement only if missing):
  [NEW-1] RD computation and maxRD export to per_seed/per_graph CSVs
  [NEW-2] CSV computation consistent with Paper 1/2
  [NEW-3] “clean floor” tagging (low-clean graphs flagged/excluded from main claim)
  [NEW-4] report.txt includes all metrics and explicitly warns against
          interpreting max_drop alone.

================================================================================
7) ONE COMMAND “RUN PLAN” (OPERATOR CHECKLIST)
----------------------------------------------
1) Stage 0/1:
   - run_plot2_topology_study.py with:
       --scale mini
       --subjects 1,3,4
       --generator_mode ws_flex
       --primary_perturbation_type ar1_drift
       --target_snr_db -12
       --alpha_grid 0,0.25,0.5,0.75,1.0
       --coverage_level regime_cl_bins_fixed
       --proxy_objectives TE,sigma
       --B_mini 8
       --S 1
       --no_modular
2) Stage 2:
   - confirm all jobs completed (no failures)
3) Stage 3:
   - analyze_plot2_results.py on the run directory
   - confirm outputs include: clean ROC-AUC, AUPC, RD/maxRD, CSV, max_drop

STOP RULE:
- Do not launch any full-scale run until Stage 3 GO criteria are satisfied.

================================================================================
END SPEC
================================================================================
