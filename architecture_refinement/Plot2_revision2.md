================================================================================
Plot 2 — Final Core Experimental Plan (Locked Specification)
================================================================================

Purpose
-------
This document defines the single, correct experimental structure for Plot 2.
Its goal is to conclusively validate whether TE and ORC can be used as
training-free proxy metrics to discover robust WS-Flex graph topologies,
without experimental contamination or baseline collapse.

This plan replaces all prior draft variants. It is intended to be executed
once, end-to-end, after staged go/no-go validation.

--------------------------------------------------------------------------------
Why This Patch Is Necessary
--------------------------------------------------------------------------------

Observations from prior runs:
- TPE-selected and “random-selected” graphs collapsed onto the same set.
- Selection overlap diagnostics showed >50–80% overlap across baselines.
- Robustness differences were weak or inconsistent under i.i.d. Gaussian noise.
- Mini runs demonstrated that CfC neuron-level dynamics can mask graph-level
  effects unless perturbations are sufficiently structured.
- Selecting “top random” graphs by TE/ORC eliminated the causal asymmetry
  required to test proxy validity.

Core issue:
- The experiment was effectively comparing “best-by-proxy vs best-by-proxy,”
  rather than proxy-guided discovery vs unguided sampling.

This patch corrects that error and restores a falsifiable experimental design.

--------------------------------------------------------------------------------
What Must Change (Non-Negotiable)
--------------------------------------------------------------------------------

CHANGE 1 — True Random Baseline
- Baseline A MUST NOT use TE or ORC for selection.
- Graphs are selected uniformly at random within structural bins.
- Only validity constraints (connectivity, degree bounds) are allowed.

CHANGE 2 — Three Distinct WS-Flex Baselines
- A: True Random (no proxy use)
- B: Random + Offline Proxy Filtering
- C: TPE (Adaptive Proxy-Guided Search)

All three share identical:
- WS-Flex generator bounds
- Training-free budget (N graphs evaluated)
- Training budget (B graphs trained)
- Structural bin constraints (degree, clustering, path length)

CHANGE 3 — Diversity Without Information Leakage
- Structural diversity (degree regime, C/L bins) is enforced.
- Proxy information (TE/ORC) is ONLY allowed in B and C.

CHANGE 4 — Overlap Gate Fix
- A–B overlap may be high (shared pool), but A must not be proxy-ranked.
- Enforce:
    overlap(A, C) < 50%
    overlap(B, C) < 50%
- Overlap is measured by stable graph identity (hash or parameters).

CHANGE 5 — Lock Perturbation Type
- Do NOT mix perturbations in Plot 2.
- Use the perturbation that produces discriminative degradation:
    AR(1) temporally correlated noise
    Target SNR = −5 dB (or validated equivalent)
- Fixed intensity grid derived from SNR mapping.

--------------------------------------------------------------------------------
What Stays Fixed
--------------------------------------------------------------------------------

Dataset:
- BNCI2014_001

Evaluation:
- Cross-session only

Model:
- Fixed CNN–WiredCfCMin (or equivalent minimal wired CfC)
- Hidden units H = 32 (locked)

Generator validity:
- nx.is_connected(G) required
- Degree k even, within [2, H−2]

Structural regimes:
- Degree regimes: super-sparse, sparse, moderate, near-dense
- Clustering / path-length used ONLY for binning, not pruning

Metrics:
- Primary robustness metric: max_drop
- Secondary: AUPC, mid_drop (diagnostic)

--------------------------------------------------------------------------------
Experimental Structure
--------------------------------------------------------------------------------

Definitions:
- N = training-free budget (graphs evaluated for TE/ORC/C/L)
- B = trained graphs per baseline

Recommended defaults:
- N = 500
- B = 12 total (3 per degree regime)
  (Minimum acceptable: B = 8, 2 per regime)

--------------------------------------------------------------------------------
Stage 0 — Training-Free Generator Diagnostic (MANDATORY)
--------------------------------------------------------------------------------

Inputs:
- Random WS-Flex pool (N graphs)
- TPE WS-Flex pool (N trials)

Outputs:
- random_ws_flex_metrics.csv
- tpe_ws_flex_metrics.csv
- Coverage summaries for degree × (C,L) bins

GO/NO-GO:
GO if:
- At least 2 regimes have ≥5/9 (C,L) bins populated in the random pool, OR
- TPE and random pools show visibly different (C,L) occupancy patterns

NO-GO:
- Severe collapse of (C,L) coverage in both pools
  → expand generator bounds before training anything

--------------------------------------------------------------------------------
Stage 1 — Candidate Set Construction (MANDATORY)
--------------------------------------------------------------------------------

Baseline A — True Random WS-Flex
- Source: random_ws_flex_metrics.csv
- Selection:
    1) Stratify by degree regime
    2) Enforce (C,L) bin coverage
    3) Uniform random selection within bins
- NO proxy ranking
- Output: selected_A.jsonl

Baseline B — Random + Offline Proxy Filtering
- Source: same random pool
- Selection:
    1) Stratify by degree regime
    2) Enforce (C,L) bin coverage
    3) Rank within bins by z_bin(TE) + z_bin(ORC)
- Output: selected_B.jsonl

Baseline C — TPE (Adaptive Proxy-Guided)
- Source: tpe_ws_flex_metrics.csv
- Selection:
    1) Stratify by degree regime
    2) Enforce (C,L) bin coverage
    3) Rank within bins by z_bin(TE) + z_bin(ORC)
- Output: selected_C.jsonl

(Optional) Baseline D — External Random Wiring
- ncps random wiring
- Density-matched to WS-Flex regimes
- Output: selected_D.jsonl

GATES:
- Exact regime counts per baseline
- collapse_score ≤ 0.50 per regime
- overlap(A, C) < 50%
- overlap(B, C) < 50%

--------------------------------------------------------------------------------
Stage 2 — Intermediate Stress Confirmation Run (FAST)
--------------------------------------------------------------------------------

Purpose:
- Verify perturbation is damaging and topology-sensitive before full training.

Setup:
- Subjects: 2–3
- Seeds: 1
- Models trained:
    Baseline A (B graphs)
    Baseline B (B graphs)
    + one fixed wiring sanity model
- Perturbation: AR(1) drift at locked SNR

GO/NO-GO:
GO if:
- max_drop ≥ 0.10 for at least one graph
- max_pairwise_delta(max_drop) ≥ 0.03 across graphs

NO-GO:
- Adjust perturbation parameters (not selection logic)

Implementation: Run architecture_refinement/run_plot2_stage2_stress.py with an
existing plot2_dir (after Stage 1). It runs the minimal job set and exits 0 (GO)
or 1 (NO-GO) based on the criteria above.

--------------------------------------------------------------------------------
Stage 3 — Full Plot 2 Run (FINAL)
--------------------------------------------------------------------------------

Proceed only if Stage 2 passes.

Setup:
- Full BNCI2014_001 cross-session
- Seeds: 2–3
- Train and evaluate:
    Baseline A
    Baseline B
    Baseline C
    (Optional D)
- Same perturbation protocol

Primary comparisons:
- Proxy validity: B vs A
- Adaptive search value: C vs B
- Out-of-family contrast (optional): A/B/C vs D

--------------------------------------------------------------------------------
Implementation Checklist
--------------------------------------------------------------------------------

Updates to existing code:
- Selection diagnostic:
    * Implement true-random selection for Baseline A
    * Update overlap gates (ignore A–B; enforce A–C, B–C)
- Selection writers:
    * Emit selected_A/B/C manifests with stable IDs

New mechanisms (if missing):
- (C,L) binning utilities per regime
- Within-bin z-score normalization for TE and ORC
- Graph identity hashing for overlap detection

Do NOT modify:
- Perturbation implementation
- Training runner
- Metric extraction logic

Note: For the locked Plot 2 core run, perturbation is fixed to AR(1) drift at −5 dB only; other perturbations are for ablation or separate analyses.

--------------------------------------------------------------------------------
End of Locked Plot 2 Core Experimental Plan
================================================================================
