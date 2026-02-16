================================================================================
Plot 2 — Pipeline Enhancements (Steps A–D) + Perturbation “Most-Damaging” Diagnostic
(Selected 3rd proxy: Spectral Radius)
================================================================================

Context / Motivation
--------------------
Recent Stage 3 results show all candidate families remain fairly robust (AUPC ~ 0.6),
which is already strong robustness. This implies the current perturbation regime may
not be sufficiently stressful or topology-sensitive to separate graph variants, or that
CfC dynamics dominate topology (topology is second-order). Before repeating any
extended training runs, we will (1) enhance the proxy search space to enable richer but
compact structures, (2) add a third topology proxy that can discriminate graphs at fixed
sparsity, and (3) perform a fast diagnostic to identify the most damaging perturbation
settings and/or distribution shifts.

This spec defines concrete changes for Steps A–D and introduces a “Most-Damaging
Perturbation Diagnostic” that must pass GO criteria before any full re-run.

================================================================================
Step A — Expand the Graph Space (Richer but Compact) Without Muddying the Story
================================================================================

Goal:
- Increase structural diversity (beyond degree/density) while keeping the narrative clean:
  “We search a *single parametric generator family* capable of expressing multiple regimes
   (lattice-like, small-world, modular, near-random) under strict capacity constraints.”

Recommendation (least muddy):
- Keep a single “WS-Flex+” generator, but add *optional internal structure knobs* so we
  do NOT introduce an entirely different family in the mainline results.

WS-Flex+ (Single Family, Extended):
1) Base WS-Flex knobs (existing):
   - H (units), k (even), p (rewiring), connectivity constraint

2) NEW optional “modularization” knob inside WS-Flex+ (implemented as a mode flag):
   - mode ∈ {plain_ws_flex, modular_ws_flex}
   - modular_ws_flex construction:
       a) Partition nodes into M modules (fixed M=4 default for H=32, equal sizes).
       b) Build WS-Flex edges *within modules* using (k_in, p_in).
       c) Add sparse inter-module edges using (k_out, p_out) or a fixed interconnect ratio r.
       d) Enforce global connectivity (repair if needed or resample).
   - This yields compact graphs with:
       - short paths (if interconnect exists),
       - richer clustering patterns,
       - controllable mixing vs compartmentalization.

Defaults (H=32):
- M = 4 modules of size 8
- k_out = 2 (or r_out = 0.05 of within-module edges)
- p_in sampled (0..1), p_out sampled (0..1) but biased low by default (e.g., 0..0.3)
- k_in sampled by existing degree regimes (super_sparse..near_dense) but applied per module

Changes to Existing Functionality:
- UPDATE: graph_generator.py (WS-Flex constructor)
  - Add generator argument: generator_mode
  - Add modular parameters: M, k_out (or r_out), p_out
  - Add stable graph identity fields to metrics output: generator_mode, M, k_in, k_out, p_in, p_out

New Mechanisms to Implement:
- NEW: build_modular_ws_flex_graph(H, M, k_in, p_in, k_out/r_out, p_out, seed)
- NEW: optional connectivity “repair” (add minimal inter-module edges) OR resample loop


GO / NO-GO for Step A (Training-Free):
- GO if the pooled graphs (N≥500) show materially broader spread in:
    - clustering C and path length L (bin coverage ≥ 6/9 in ≥2 regimes),
    - spectral radius ρ(A) range is non-trivial (e.g., IQR not collapsing to ~0),
  compared to plain WS-Flex.
- NO-GO if modular mode collapses to near-identical C/L/ρ distributions or fails feasibility.


================================================================================
Step B — Add Third Proxy Metric: Spectral Radius ρ(A)
================================================================================

Goal:
- Add a proxy that is plausibly related to dynamical amplification/stability propensity
  and can differentiate graphs at fixed sparsity.

Definition:
- Let A be the (0/1) adjacency matrix of the graph (use directed if wiring is directed;
  otherwise symmetric for undirected).
- Spectral radius proxy:
    ρ(A) = max_i |λ_i(A)|
  where λ_i are eigenvalues of A.

Normalization (for selection/optimization):
- Because ρ(A) increases with degree and graph size, normalize within bins to avoid
  degenerating into “choose dense graphs”.
- Use within-(regime × (C,L) bin) robust z-score:
    z_ρ = (ρ - median(ρ_bin)) / (MAD(ρ_bin) + ε)
- For multi-objective optimization, either:
    (a) maximize z_ρ, or
    (b) target a “sweet spot” interval for stability propensity:
        minimize |z_ρ - z_target| with z_target = 0 (default) or tuned later
  (Start with (a) maximize z_ρ ONLY for diagnostics; for final search, prefer (b)
   to avoid always pushing toward maximal amplification.)

Pipeline Changes:
- UPDATE: topology_analyzer.py
  - Compute ρ(A) for each accepted graph and write field: spectral_radius
- UPDATE: metrics CSV schemas (random_ws_flex_metrics.csv, tpe_ws_flex_metrics.csv)
  - Add columns: spectral_radius, spectral_radius_norm (optional), z_rho_bin (optional)

New Mechanisms to Implement:
- NEW: compute_spectral_radius(A) utility
  - Use numpy.linalg.eigvals for H=32 (cheap enough)
  - Cache by graph hash to avoid recomputation

GO / NO-GO for Step B:
- GO if ρ(A) exhibits meaningful variability within capacity-controlled subsets.
- NO-GO if ρ(A) is almost perfectly monotonic with degree/density inside your bins
  (then it adds little beyond TE/ORC).


================================================================================
Step C — Strict Capacity Control (Compactness Must Be Enforced)
================================================================================

Goal:
- Prevent “wins” due to larger capacity rather than topology.
- Allow limited capacity variation only within locked bounds.

Capacity Controls (Locked for next run):
- Hidden units: H ∈ {24, 32} (two-point sweep) OR keep H=32 fixed (preferred next step).
- Active edges constraint: enforce E_active in a fixed band per regime:
    super_sparse:  E_active ∈ [E1_min, E1_max]
    sparse:        E_active ∈ [E2_min, E2_max]
    moderate:      E_active ∈ [E3_min, E3_max]
    near_dense:    E_active ∈ [E4_min, E4_max]
  (These bands are derived directly from k ranges at H=32; ensure consistency across modes.)

Implementation Notes (masked-weight wiring):
- You must report BOTH:
    - H
    - E_active (mask active edges)
  because dense weights exist but are masked.

Pipeline Changes:
- UPDATE: generator feasibility/acceptance gate
  - Add capacity filter: reject graphs whose derived E_active falls outside regime band.
- UPDATE: selection diagnostics
  - Ensure density bands are matched between baselines and between generator_mode.

New Mechanisms to Implement:
- NEW: compute_E_active_from_graph(G, wiring_build) and enforce bands
- NEW: “capacity manifest” written alongside selected architectures

GO / NO-GO for Step C:
- GO if the selected A/B/C candidate sets have matched distributions of:
    - H (if varied)
    - E_active and mask_density
- NO-GO if any baseline systematically gets higher E_active or larger H.


================================================================================
Step D — Make Perturbations Topology-Sensitive (Increase Stress / Add Shift)
================================================================================

Goal:
- Achieve meaningful separation (not all AUPC ~0.6) without unrealistic artifacts.
- Use perturbations that target temporal structure and distribution shift.

Primary perturbation (keep):
- AR(1) drift (temporal correlation) with target SNR in dB

Enhancements (choose 1–2, not many):
1) Increase intensity:
   - expand SNR grid downward (more severe):
       SNR_dB ∈ {0, -3, -6, -9, -12}  (default)
   - OR keep target at -5 but increase drift persistence:
       lag1_autocorr target from ~0.97 → ~0.99 (default optional)

2) Add distribution shift (recommended; topology-sensitive):
   - “session shift” augmentation at test time:
       - per-channel gain drift (slow multiplicative drift),
       - baseline offset drift (slow additive drift),
       - small temporal jitter of windows (±Δt ms)
   - These are plausible acquisition-induced shifts.

3) Spatially structured dropout (secondary; topology-sensitive):
   - contiguous channel-region dropout (simulated electrode cluster failure)
   - use as secondary perturbation or appendix only

Pipeline Changes:
- UPDATE: perturbation module(s) to accept:
   - extended SNR grid
   - optional drift persistence target
   - optional shift knobs (gain drift, offset drift, jitter)
- UPDATE: perturbation fingerprint to include these new knobs

GO / NO-GO for Step D (via diagnostic below):
- GO if at max intensity:
    - max_drop ≥ 0.15 for at least one baseline graph
    - max_pairwise_delta(max_drop) ≥ 0.05 across candidate graphs
- NO-GO if degradation saturates weakly (still AUPC high and deltas small)
  → increase severity or switch to shift-based perturbation.


================================================================================
MOST-DAMAGING PERTURBATION DIAGNOSTIC (MANDATORY BEFORE FULL TRAINING)
================================================================================

Purpose:
- Rapidly identify which perturbation type/settings produce the strongest and most
  topology-sensitive degradation, so we do not waste another 24-hour run.

Approach:
- Use a 1-model (or 2-model) diagnostic with tiny compute:
  - 1 subject (or 2 subjects)
  - 1 seed
  - 4 graphs spanning regimes (super_sparse, sparse, moderate, near_dense)
  - no TPE vs random comparison yet (just sensitivity)

Candidate perturbations to sweep (in order):
A) AR(1) drift SNR grid: {0, -3, -6, -9, -12}
B) AR(1)+gain drift (multiplicative): small→large
C) AR(1)+offset drift (additive): small→large
D) Temporal jitter: ±{0, 10, 20, 40} ms
E) Spatial dropout (clustered): {small, medium, large cluster}

Outputs:
- For each perturbation setting:
   - clean ROC-AUC
   - max_drop
   - AUPC
   - topology sensitivity score:
       S_topo = max_pairwise_delta(max_drop) across the 4 graphs

GO / NO-GO (Diagnostic Pass Criteria):
- Select the perturbation configuration P* that maximizes S_topo, subject to:
    1) max_drop_max(P*) ≥ 0.15  (it is actually damaging)
    2) S_topo(P*) ≥ 0.05       (it differentiates topology)
- If no configuration satisfies both:
    - Increase severity range (e.g., SNR down to -15 dB) OR
    - Prefer shift-based perturbations (gain/offset/jitter) over pure noise.

Integration / Reuse of Existing Scripts:
- ADAPT: existing smoke-test runner (Phase 5-style) to:
   - loop over perturbation configs
   - train once per graph (or reuse trained weights if perturbation is test-time only)
   - evaluate robustness metrics per config
- REUSE: existing analyzer that computes AUPC/max_drop
- UPDATE: diagnostics runner to write a “perturbation sweep report.json” + table

New Mechanisms to Implement:
- NEW: perturbation_sweep_config.yaml + loop driver script:
    run_plot2_perturbation_sweep.py
- NEW: selection of P* and auto-write “locked_perturbation_config.yaml”

Decision Points (Defaults):
- Keep single perturbation type for the next full run:
    default: AR(1) drift with extended SNR grid down to -12 dB
- Only add one distribution shift knob if needed:
    default add-on: gain drift (multiplicative), mild→moderate

--------------------------------------------------------------------------------
Execution Order (Do Not Deviate)
--------------------------------------------------------------------------------

1) Step A training-free check: WS-Flex+ generator feasibility and diversity
2) Step B training-free check: compute spectral radius ρ(A) and verify non-collapse
3) Step C enforce capacity controls and verify matched E_active across pools
4) Run Most-Damaging Perturbation Diagnostic and lock perturbation config P*
5) ONLY THEN run the full Plot 2 A/B/C experiment with locked P* and capacity controls

--------------------------------------------------------------------------------
End of Spec
================================================================================
