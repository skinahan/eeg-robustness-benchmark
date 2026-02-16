====================================================================
Plot 2 — Revised Experiment Specification (Baseline A/B/C Framework)
====================================================================

This specification revises Plot 2 to correct a conceptual flaw identified
during the mini-scale experiments and to align the experimental logic
with the *actual claim* of the paper:

    “Topological Entropy (TE) and Ollivier–Ricci Curvature (ORC) can be
     used as *proxy metrics* to efficiently identify robust recurrent
     graph topologies, inverting the empirical correlation analysis of
     Waqas et al. into a training-free search procedure.”

The key change is a principled separation between:
- **proxy usefulness**, and
- **adaptive proxy-guided optimization**.

This is achieved by introducing three distinct WS-Flex baselines
(A/B/C), each with identical compute budgets but different access to
proxy information.

====================================================================
Motivation and Gap Relative to Prior Work
====================================================================

Waqas et al. (2022) and You et al. (2020) establish that:
- Graph metrics such as entropy, curvature, clustering, and path length
  correlate with robustness *after* model training.
- Their methodology samples large graph sets, trains models, and then
  analyzes correlations retrospectively.

Our work seeks to *invert this process*:
- Use TE and ORC **before training** to guide graph selection.
- Avoid training thousands of models.
- Convert selected graphs into structured CfC/NCP-style recurrent
  architectures.

However, earlier Plot 2 iterations selected *both* “random” and “TPE”
graphs using TE/ORC, collapsing the comparison into:
    “best TE+ORC vs best TE+ORC.”

This obscures whether:
(a) the proxies themselves are useful, and/or
(b) adaptive search (TPE) improves efficiency.

The revised Plot 2 explicitly disentangles these effects.

====================================================================
Core Experimental Claim Structure
====================================================================

Plot 2 now answers three nested questions:

Q1 (Proxy validity):
    Do graphs selected using TE/ORC exhibit greater robustness than
    truly random graphs from the same generator family?

Q2 (Search efficiency):
    Given TE/ORC as proxies, does adaptive sampling (TPE) identify
    robust graphs more efficiently than offline proxy filtering?

Q3 (Structural bias):
    Does the WS-Flex family (with proxy guidance) outperform generic
    random wiring not constrained to WS-Flex?

These correspond directly to Baselines A, B, and C below.

====================================================================
Baseline Definitions (Locked)
====================================================================

All baselines:
- Use the same WS-Flex generator bounds (k, p, connectivity-only).
- Use identical robustness evaluation, perturbations, and metrics.
- Train the same number of models B.
- Respect identical regime and coverage constraints.

The only difference is **how candidate graphs are chosen**.

------------------------------------------------------------
Baseline A — True Random WS-Flex (No Proxy Selection)
------------------------------------------------------------

Purpose:
- Establish the *natural robustness distribution* of the WS-Flex graph
  family under the given constraints.

Procedure:
1) Sample N graphs uniformly from WS-Flex:
   - k sampled by degree regimes
   - p ~ Uniform(0, 1)
   - connectivity enforced
2) Compute topology metrics (TE, ORC, C, L) for logging only.
3) Select B graphs **uniformly at random**, subject only to:
   - degree-regime stratification
   - (C, L) coverage bin constraints (see Selection Policy)
4) Train and evaluate these B graphs.

Important:
- TE and ORC are NOT used for ranking or selection.
- This is the correct baseline for testing proxy usefulness.

------------------------------------------------------------
Baseline B — Random WS-Flex + Offline Proxy Filtering
------------------------------------------------------------

Purpose:
- Test whether TE/ORC are *predictive* of robustness, even without an
  adaptive optimizer.

Procedure:
1) Sample the same N graphs uniformly from WS-Flex as in Baseline A.
2) Compute TE and ORC for all N graphs.
3) Select B graphs using:
   - degree-regime stratification
   - (C, L) coverage bins
   - within-bin ranking by proxy score:
         score(G) = z_bin(TE(G)) + z_bin(ORC(G))
4) Train and evaluate these B graphs.

Interpretation:
- Improvement of Baseline B over A demonstrates that TE/ORC are useful
  robustness proxies (inverting Waqas et al.’s correlation analysis).

------------------------------------------------------------
Baseline C — TPE (Adaptive Proxy-Guided WS-Flex Search)
------------------------------------------------------------

Purpose:
- Test whether *adaptive* proxy-guided search improves sample efficiency
  beyond offline proxy filtering.

Procedure:
1) Run Optuna TPE for N trials on WS-Flex parameters.
   - Objectives: maximize TE, maximize ORC
2) Retain all evaluated trials.
3) Select B graphs from TPE trials using the SAME selection policy as
   Baseline B:
   - regime stratification
   - (C, L) coverage bins
   - within-bin proxy ranking
4) Train and evaluate these B graphs.

Interpretation:
- Improvement of Baseline C over B demonstrates that adaptive sampling
  is beneficial beyond static proxy filtering.
- If C ≈ B, the proxy itself is the dominant contribution.

------------------------------------------------------------
Additional Comparators (Unchanged)
------------------------------------------------------------

Baseline D — External Random Wiring (Out-of-Family)
- Uses ncps random wiring class (or equivalent).
- Density-matched selection only.
- Tests whether WS-Flex itself is a strong inductive bias.

Baseline E — Hand-Designed Baseline (Capacity-Matched)
- Fixed wiring (e.g., AutoNCP or canonical NCP).
- Hidden units and effective capacity explicitly matched to WS-Flex.
- Serves as a reference, not a competitor in proxy claims.

====================================================================
Budgets and Fairness Constraints (Locked)
====================================================================

- Training-free budget N:
    Same for Baselines A, B, and C (e.g., N = 250 or 500).
- Trained models B:
    Same across A/B/C (e.g., B = 12 total, 3 per degree regime).
- Compute fairness:
    All trained models must satisfy the same capacity constraints
    (hidden units, mask density ranges, FLOPS proxy tolerance).

No baseline receives:
- Additional proxy information
- Additional training runs
- Additional selection degrees of freedom

====================================================================
Selection Policy (Shared Across A/B/C)
====================================================================

Selection constraints applied AFTER candidate pool creation:

1) Degree regime stratification (super-sparse → near-dense).
2) (C, L) coverage binning within each regime:
   - C bins: tertiles
   - L bins: tertiles
3) Collapse constraint:
   - No bin may contain >50% of selected graphs per regime.
4) Ranking:
   - Baseline A: uniform random within bins
   - Baselines B/C: proxy ranking within bins

This ensures diversity without contaminating the baseline definitions.

====================================================================
Primary Metrics and Claims (Locked)
====================================================================

Primary robustness metric:
- Worst-case degradation:
      max_drop = ROC_AUC(clean) − ROC_AUC(max perturbation)

Secondary metrics:
- AUPC
- mid_drop (optional diagnostic)

Primary statistical comparisons:
1) B − A : proxy validity
2) C − B : adaptive search benefit
3) B/C − D : WS-Flex vs generic random wiring

====================================================================
Expected Interpretations (All Acceptable Outcomes)
====================================================================

- If B > A:
    TE/ORC are validated as useful proxies.
- If C > B:
    Adaptive TPE improves sample efficiency.
- If C ≈ B > A:
    Proxies dominate; TPE is optional.
- If B ≈ A but B/C > D:
    WS-Flex bias + constraints matter more than proxy ranking.
- If all ≈:
    Robustness arises primarily from CfC dynamics; topology has limited
    influence under this perturbation.

All outcomes are interpretable and scientifically valid.

====================================================================
Narrative Alignment for the Paper
====================================================================

Plot 2 is explicitly framed as:
- An inversion of Waqas et al.’s post-hoc robustness correlations.
- A test of whether those correlations can be exploited *a priori*.
- A study of proxy-guided architectural search efficiency, not just
  final performance.

This revised structure prevents baseline contamination, clarifies
causal claims, and ensures reviewer-proof logic.

====================================================================
Plot 2 — Revised Experiment Specification with Staged Next Steps
(A/B/C Baselines + Go / No-Go Gates)
====================================================================

This document extends the revised Plot 2 specification by explicitly
defining the **required staged steps that must precede full-scale
training**, along with **clear go / no-go criteria** at each stage.

The intent is to:
- prevent unnecessary 24-hour training runs,
- diagnose whether limitations arise from the generator, proxies,
  selection policy, or model dynamics,
- and ensure that any full-scale Plot 2 result is interpretable,
  defensible, and reviewer-proof.

====================================================================
High-Level Logic of the Staging
====================================================================

Plot 2 proceeds through four ordered stages:

Stage 0 — Generator & Measure-Space Diagnostics (training-free)
Stage 1 — Selection Policy Validation (training-free)
Stage 2 — Proxy Usefulness Mini-Run (small-scale training)
Stage 3 — Adaptive Search Mini-Run (small-scale training)
Stage 4 — Full-Scale Plot 2 (only if all prior gates pass)

No stage may be skipped.
Failure at any stage requires revision before proceeding.

====================================================================
Stage 0 — Generator & Measure-Space Diagnostic (MANDATORY)
====================================================================

Purpose:
- Determine whether the WS-Flex generator (k, p, connectivity-only)
  actually spans meaningful diversity in graph-measure space.
- Identify whether observed saturation is due to generator collapse
  or downstream selection/training effects.

Procedure:
- Generate:
    - N_random = 2000 WS-Flex graphs (uniform sampling)
    - N_tpe = 2000 WS-Flex graphs via TPE (TE, ORC objectives)
- Compute per-graph metrics:
    k, density, C, L, TE, ORC
- No selection, no training.

Required analyses:
- Distribution plots of C and L per degree regime
- (C, L) scatter plots per regime
- Coverage scores:
    coverage = (# occupied (C_bin, L_bin)) / 9
- Correlation matrices:
    corr(k, ORC), corr(k, C), corr(k, L), corr(ORC, C), corr(ORC, L)

GO / NO-GO CRITERIA:
GO if at least one of the following holds:
- coverage ≥ 0.6 in ≥2 degree regimes (random or TPE), OR
- TPE shows statistically distinct (C, L) occupancy vs random
  (e.g., KS test p < 0.05 for C or L distributions)

NO-GO if:
- coverage ≤ 0.4 in most regimes for both random and TPE, AND
- TPE and random (C, L) distributions are nearly identical

If NO-GO:
- STOP.
- Expand the search space (e.g., introduce soft C/L objectives,
  additional WS-Flex controls, or directed edges).
- Do NOT proceed to Stage 1.

====================================================================
Stage 1 — Coverage-Aware Selection Validation (MANDATORY)
====================================================================

Purpose:
- Verify that the new selection policy enforces structural diversity
  without biasing toward any single regime or bin.
- Ensure that “random” and “proxy-guided” sets are meaningfully distinct
  before any training occurs.

Procedure:
- Using the pools from Stage 0:
    - Apply coverage-aware selection to:
        Baseline A pool (uniform-in-bin)
        Baseline B pool (proxy-ranked-in-bin)
        Baseline C pool (TPE trials, proxy-ranked-in-bin)
- Select B graphs per method (e.g., B = 12).

Required diagnostics:
- Regime counts
- (C_bin, L_bin) counts
- collapse_score
- coverage_score

GO / NO-GO CRITERIA:
GO if all of the following hold for A, B, and C:
- collapse_score ≤ 0.50
- coverage_score ≥ 0.5 overall
- Selected sets for A, B, and C are not identical
  (e.g., <50% graph overlap by hash/parameters)

NO-GO if:
- Selection collapses into the same bins or graphs across methods, OR
- Coverage constraints cannot be satisfied without heavy fallback

If NO-GO:
- STOP.
- Revise selection bins or regime definitions.
- Do NOT proceed to training.

====================================================================
Stage 2 — Proxy Usefulness Mini-Run (Baseline A vs B)
====================================================================

Purpose:
- Test the *core hypothesis* that TE/ORC are useful robustness proxies,
  independent of adaptive search.

Baselines involved:
- Baseline A: True Random WS-Flex
- Baseline B: Random WS-Flex + Offline Proxy Filtering
- (Baseline D: External Random Wiring, optional but recommended)

Procedure:
- Subjects: 2–3 (minimum)
- Seeds: 1
- Trained models per baseline: B (e.g., 12)
- Perturbation: AR(1) drift, target SNR = −5 dB
- Metrics: clean ROC-AUC, max_drop, AUPC

GO / NO-GO CRITERIA:
GO if:
- Mean(max_drop_B − max_drop_A) < 0 (proxy-filtered is more robust), AND
- Effect is directionally consistent across ≥2 subjects, AND
- 95% bootstrap CI width for (B − A) ≤ 0.10

NO-GO if:
- Baseline B ≈ Baseline A on max_drop and AUPC, OR
- Effects are inconsistent in sign across subjects

If NO-GO:
- STOP.
- Reconsider proxy choice (TE/ORC), perturbation type, or claim scope.
- Do NOT proceed to adaptive search.

====================================================================
Stage 3 — Adaptive Search Mini-Run (Baseline B vs C)
====================================================================

Purpose:
- Determine whether adaptive proxy-guided search (TPE) improves sample
  efficiency beyond offline proxy filtering.

Baselines involved:
- Baseline B: Random + Offline Proxy Filtering
- Baseline C: TPE (Adaptive Proxy-Guided)

Procedure:
- Same subjects, seeds, perturbation, and metrics as Stage 2.
- Same training budget B.
- Same coverage-aware selection policy.

GO / NO-GO CRITERIA:
GO if at least one holds:
- Mean(max_drop_C − max_drop_B) < 0 with CI not absurdly wide, OR
- C reaches comparable robustness using a *smaller effective N*
  (demonstrated via ablation or subsampling)

NO-GO if:
- Baseline C ≈ Baseline B with no efficiency or robustness advantage

Interpretation if NO-GO:
- Proxy usefulness is validated, but TPE is optional.
- Paper should emphasize proxy-to-wiring pipeline rather than NAS.

====================================================================
Stage 4 — Full-Scale Plot 2 (FINAL)
====================================================================

This stage is permitted ONLY if:
- Stage 2 passes (proxy usefulness established), AND
- Stage 1 and Stage 0 passed integrity and coverage gates.

Procedure:
- Subjects: full BNCI2014_001 cross-session set
- Seeds: ≥2 (preferably 3)
- Baselines: A, B, C, D (+ baseline E as reference)
- Metrics: primary = max_drop; secondary = AUPC

Required reporting:
- B − A (proxy validity)
- C − B (adaptive benefit, if present)
- B/C − D (WS-Flex vs generic random wiring)

====================================================================
Rationale for the Staging
====================================================================

This staged design ensures that:
- Each claim is tested in isolation.
- Failures are informative, not ambiguous.
- Expensive training is only performed when logically justified.
- The final Plot 2 narrative directly supports the paper’s central
  contribution: *inverting post-hoc robustness correlations into a
  training-free graph selection pipeline*.

====================================================================
End of Staged Plot 2 Specification
====================================================================

