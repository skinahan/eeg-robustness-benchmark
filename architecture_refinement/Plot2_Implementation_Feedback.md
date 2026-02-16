================================================================================
PLOT 2 IMPLEMENTATION REVIEW — RISKS, DIVERGENCES, AND PATCH RECOMMENDATIONS
(Extremely critical pass; focused on best practices, NeurIPS checklist alignment,
and consistency with our staged experimental plan.)
================================================================================

Scope
-----
This review is based ONLY on the pasted “PLOT 2 IMPLEMENTATION NOTES”. I am
treating that document as the operational truth and flagging: (i) statements
that overclaim, (ii) implementation choices that can bias results or reduce
interpretability, and (iii) gaps vs. ML best practices / NeurIPS checklist.

--------------------------------------------------------------------------------
A. CLAIMS / FRAMING PROBLEMS (must be toned down in any report)
--------------------------------------------------------------------------------

A1) “Causal evidence that wiring topology alone drives robustness differences”
    Problem:
    - The current design does not justify a “topology alone” causal claim unless:
      (i) capacity is strictly matched across *all* baselines (including NCP and
      external random), (ii) training protocol is identical and locked, (iii)
      the only manipulated variable is topology, and (iv) you demonstrate that
      selection and search do not introduce confounds.
    Risk:
    - Reviewers will reject “causal” language without stronger controls.
    Patch:
    - Replace with “controlled comparison under matched protocol” + explicitly
      list what is held constant and what is allowed to vary (H, E_active, etc.).

A2) “NAS-selected wiring yields higher robustness than random wiring…”
    Problem:
    - Implementation describes multiple “random” baselines and (in analysis) a
      mixed “Random set” that includes baseline_b / random_stratified in some
      paths. This can blur what “random” means and undermines clean inference.
    Patch:
    - Lock A/B/C semantics and ensure analysis does NOT pool dissimilar baselines.

--------------------------------------------------------------------------------
B. EXPERIMENTAL DESIGN RISKS / POTENTIAL CONFOUNDS
--------------------------------------------------------------------------------

B1) Capacity control is not fully guaranteed across all baselines
    Observed:
    - WS-Flex candidates: capacity_filter uses E_active = H*k/2 (OK for undirected WS).
    - NCP baseline: “units=H+ncp_io_size” implies additional units vs H=32 recurrent
      chamber, depending on how ncp_io_size is defined.
    - External random: uses ncps.wirings.Random(units=output_size, output_dim=output_size,
      sparsity_level=sp). This seems inconsistent with H=32. If units != H (or if “output_size”
      differs), then you are not capacity-matching.
    Risk:
    - Baseline “wins” can be explained by different effective state size or different mask
      density ranges. Reviewers will immediately question this.
    Patch (required):
    - Define ONE capacity schema for ALL baselines:
        * recurrent hidden units H_fixed (e.g., 32)
        * active edges E_active (or mask_density) within regime bands
      And enforce:
        * NCP baseline uses exactly H_fixed in the recurrent chamber
        * external random uses units=H_fixed (not output_size) and is stratified by E_active

B2) Edge orientation policy may introduce uncontrolled variability
    Observed:
    - WS-Flex graphs are built undirected, then oriented via “random_oriented” in wiring.
    - Orientation is seed-dependent (wiring_seed) and could be a major driver of dynamics.
    Risk:
    - You may be inadvertently comparing “orientation distributions” more than topology.
    Patch:
    - Make orientation deterministic given graph_seed (or explicitly treat orientation as part
      of the “topology” and control it identically across baselines).
    - Report oriented adjacency spectral radius (directed) if that’s what the model actually uses.

B3) Selection bins (C/L tertiles) are computed per regime from the pool itself
    Observed:
    - _compute_cl_bins uses tertiles computed from the candidate pool.
    Risk:
    - This makes selection dependent on the sampled pool distribution. If the pool is small
      or biased (e.g., connectivity rejection), bin boundaries drift run-to-run.
    Patch:
    - For reproducibility: compute bin edges from a fixed “reference” sample (like M_ref),
      or store the exact bin edges in the manifest and reuse them for selection.

B4) Connectivity rejection biases the candidate distribution
    Observed:
    - Disconnected graphs are pruned (nx.is_connected).
    Risk:
    - This implicitly changes the sampling distribution over (k,p); the effective prior is
      “connected WS graphs” not “WS graphs”. That can compress diversity and distort proxy relationships.
    Patch:
    - Record rejection rates by (k,p) and consider either:
        * (preferred) resampling but logging acceptance probability as a function of (k,p), or
        * adding a “repair to connected” mechanism with explicit reporting.

B5) “Random baseline” ambiguity / pooling risk in analysis
    Observed:
    - analysis says: Random: {"random_stratified","baseline_a","baseline_b"} in AUPC path.
    Risk:
    - If baseline_b is proxy-ranked, it is NOT a random baseline. Pooling it into “Random”
      contaminates the comparator and can nullify differences (or fabricate them).
    Patch (required):
    - Hard-separate groups in analysis:
        A = true random (uniform within bins)
        B = proxy-filtered from shared pool
        C = adaptive/TPE
        NCP = hand-designed
        External = ncps Random
      Do NOT pool B with A. Do NOT pool “random_stratified” unless its definition matches A.

B6) Hypervolume early stopping can lead to “premature saturation”
    Observed:
    - HV computed on Pareto front with fixed reference bounds and ref point (-0.05,-0.05).
    Risk:
    - If TE/ORC are normalized/clipped tightly by quantiles, HV improvement will plateau quickly,
      forcing you to hit M_max or stop early without meaningfully exploring.
    Patch:
    - Store raw TE/ORC distributions and verify:
        * Are values saturating at [0,1]?
        * Are quantile bounds too tight?
      Consider using unbounded transforms (e.g., z-scores) for HV or widening bounds.

--------------------------------------------------------------------------------
C. METRICS / PROXIES — SPECIFIC TECHNICAL PITFALLS
--------------------------------------------------------------------------------

C1) Spectral radius needs to match the *actual* directed wiring used by the model
    Observed:
    - analyzer computes spectral radius (unclear if on undirected A or directed A).
    Risk:
    - If model uses oriented adjacency, undirected spectral radius can be a weak proxy.
    Patch:
    - Compute ρ(A_dir) on the directed adjacency actually used in CfC wiring.

C2) AUPC integration must have a physically meaningful x-axis
    Observed:
    - AUPC integrates metric vs intensity then divides by sigma_max.
    Risk:
    - If “intensity” is SNR dB, trapezoidal integration over dB is not physically linear.
      If “intensity” is alpha ∈ [0,1], it is okay, but you must ensure the runner outputs
      consistent alpha and that sigma_max reflects the same parameter.
    Patch:
    - Lock AUPC definition per perturbation type:
        * If driver uses alpha in [0,1], AUPC_alpha is valid.
        * If driver uses SNR_dB grid, define AUPC over alpha mapping, not raw dB.
      Add explicit check: intensity grid monotonic and in expected units, else fail.

C3) “max_drop = clean - roc(alpha=1)” assumes alpha=1 is meaningful maximum
    Observed:
    - For correlated perturbations, alpha_max may not correspond to the intended max stress
      if target_snr_db is separately enforced.
    Patch:
    - For SNR-targeted perturbations, define max_drop at the worst SNR point, not alpha=1,
      unless alpha=1 is guaranteed to equal “target_snr_db max stress”.

C4) TE/ORC normalization from quantiles risks clipping away informative extremes
    Observed:
    - reference bounds are quantiles q_lo, q_hi.
    Risk:
    - Extreme graphs beyond q_hi collapse to the same te_norm/orc_norm (if clipped), harming
      selection and HV.
    Patch:
    - Save % clipped above/below bounds. If >5–10%, widen bounds or switch to robust z-scoring.

--------------------------------------------------------------------------------
D. REPRODUCIBILITY / NEURIPS CHECKLIST ALIGNMENT
--------------------------------------------------------------------------------

D1) Reproducibility: seeds are good, but randomness sources are distributed
    Observed:
    - graph_seed, wiring_seed, training_seeds are derived; Optuna sampler has own RNG;
      community detection not present; orientation random.
    Risk:
    - “Same run_id = same results” may fail if any RNG is not locked.
    Patch:
    - Manifest must explicitly record:
        * optuna sampler seed
        * reference bounds seed
        * binning seed (if any)
        * orientation policy and seed mapping
      Add a “replay mode” that regenerates selected architectures from manifest only.

D2) Statistical best practice: hierarchical bootstrap is good, but “unit of analysis”
    must match claim
    Observed:
    - bootstrap resamples graphs then seeds within graph.
    Risk:
    - If subject variability exists, the next-level hierarchy is subject → graph → seed.
      If subjects are pooled earlier, CI may be overconfident or miscalibrated.
    Patch:
    - Ensure the bootstrap hierarchy matches the data structure actually produced:
        subject (or session) → graph → seed
      If subjects are fixed list and included, incorporate them in resampling or explicitly
      state limitation (conditional on chosen subjects).

D3) Compute/Resource reporting (NeurIPS)
    Gap:
    - Implementation writes git_commit, but you should also record:
        * GPU type(s), CPU, wall-clock per job, #params (effective active weights),
          and total compute budget.
    Patch:
    - Add “compute_manifest.json” with per-job runtime and hardware.

D4) Data leakage / split hygiene
    Unknown from notes:
    - Must confirm that cross-session evaluation is implemented correctly and that no test
      information is used during proxy selection (it shouldn’t be; proxies are training-free).
    Patch:
    - Explicitly verify and log: train sessions vs test sessions for each subject.

--------------------------------------------------------------------------------
E. DIVERGENCE FROM OUR PLAN / BEST PRACTICE
--------------------------------------------------------------------------------

E1) The pipeline still allows a subtle “proxy fairness” trap
    Observed:
    - Baseline B is proxy-filtered from random pool; C is TPE maximizing TE/ORC.
    - If you also select the “random comparator” by TE/ORC (baseline B), the comparison can
      become “best TE/ORC vs best TE/ORC” and reduce separability.
    Status:
    - This is addressed by Baseline A (true random). BUT analysis must not pool B into random.
    Required:
    - Enforce A as the primary comparator for inversion claim.

E2) External random selection uses deterministic sorting, not randomness
    Observed:
    - “deterministic sort by (mask_density, wiring_seed)” after stratification.
    Risk:
    - This is not “random” and may bias selection systematically.
    Patch:
    - Use a seeded shuffle within bins, or sample uniformly within bins (with fixed seed).

--------------------------------------------------------------------------------
F. ACTIONABLE PATCH LIST (NO AMBIGUITY)
--------------------------------------------------------------------------------

F1) Analysis grouping patch (REQUIRED)
- Remove baseline_b from “Random” pool in analyze_plot2_results.py.
- Define explicit method groups: A, B, C, NCP, External.
- Fail hard if method labels missing or pooled.

F2) Capacity alignment patch (REQUIRED)
- External random: units must equal H (32), not output_size.
- NCP baseline: recurrent chamber must be H=32 (no hidden unit advantage).
- Write and enforce E_active bands for ALL baselines; verify matched distributions.

F3) Spectral radius patch (RECOMMENDED)
- Compute ρ(A_dir) on the oriented adjacency actually used by the model.
- Store both ρ_undir and ρ_dir for diagnostics.

F4) Bin edges determinism patch (RECOMMENDED)
- Derive C/L bin edges from M_ref reference sample and store in manifest.
- Use these fixed edges for A/B/C selection.

F5) “Intensity axis” validation patch (REQUIRED for credibility)
- In analysis: validate intensity units per perturbation type.
- If intensity is SNR_dB, compute AUPC over alpha or over a linearized mapping.

F6) External random sampling patch (RECOMMENDED)
- Replace deterministic sort with seeded random selection within density bins.

F7) Bootstrap hierarchy patch (IF subjects >1 and used for inference)
- Ensure subject-level resampling is included or explicitly state conditional inference.

--------------------------------------------------------------------------------
G. FINAL NOTE ON STRENGTH OF CLAIMS (implementation-note doc is too strong)
--------------------------------------------------------------------------------

Even if everything is implemented perfectly, Plot 2 should not be described as
“causal evidence” unless you (i) fully control capacity and selection confounds,
(ii) explicitly define the intervention (topology) and the invariants, and
(iii) show sensitivity checks that rule out alternative explanations (orientation,
density, clipping artifacts, intensity axis mistakes).

Right now, the implementation notes overclaim. The pipeline can still produce
valuable evidence, but language must be reduced to “controlled comparison” unless
the above patches are applied and validated.

================================================================================
End of Review
================================================================================
