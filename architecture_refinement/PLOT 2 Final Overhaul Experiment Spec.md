PLOT 2 — DECISIVE EXPERIMENT SPEC FOR “TPE > RANDOM” (WS-FLEX) + PIVOT PLAN
Date: 2026-02-17 (America/Phoenix)
Scope: mini-scale subjects {1,3,4}; robustness under ar1_drift with target_snr_db ∈ {-12, -6}

============================================================
GLOBAL DEFINITIONS (used by all modules)
============================================================

G0. Graph objects
- Undirected hidden adjacency: A ∈ {0,1}^{H×H}, symmetric, A_ii=0
- Oriented hidden adjacency: Ã ∈ {0,1}^{H×H}, produced by a deterministic orientation rule with seed s_orient(g)
- H (hidden size): 32 (unless explicitly varied)

G1. Graph validity
- Enforce connectedness on undirected graph G(A). If disconnected:
  - Option A (preferred): resample graph
  - Option B: take largest connected component and re-index (only if allowed by model constraints)

G2. Deterministic seeds for reproducibility
- graph_seed: controls WS-Flex generation
- wiring/orientation seed: s_orient(g) = hash(graph_hash) mod 2^31-1
- training seeds: S seeds derived from run_id + fixed salt, shared across all arms

G3. Robustness evaluation grid and perturbation
- perturbation_types: ar1_drift
- alpha_grid: A = {0.0, 0.25, 0.5, 0.75, 1.0}  (optional dense: add 0.125 steps for smoother RD)
- target_snr_db: evaluate BOTH -12 and -6 using the SAME trained weights (evaluation-only delta)

G4. Performance and robustness metrics (per graph g)
Let p_g(α) = ROC-AUC at intensity α; p_clean = p_g(0)

- Absolute drop curve:
  Δ_g(α) = p_clean - p_g(α)

- max_drop (lower is better):
  max_drop(g) = max_{α∈A} Δ_g(α)

- Robustness-normalized degradation curve (controls for clean AUC):
  RD_g(α) = (p_clean - p_g(α)) / max(p_clean - 0.5, ε)
  choose ε = 1e-3

- maxRD (lower is better):
  maxRD(g) = max_{α∈A} RD_g(α)

- AUPC (higher is better):
  AUPC(g) = mean_{α∈A} p_g(α)          (or trapezoid integral if A dense)

Report per-graph seed variance:
- Var_seed(p_clean), Var_seed(maxRD)

Primary metric for decisive comparison:
- maxRD (lower is better)

Secondary:
- p_clean and Pareto analysis (p_clean vs maxRD)
- max_drop and AUPC as supporting

G5. Selection arms to compare (WS-Flex only)
- ARM_RAND: random selection baseline
- ARM_TPE: TPE-selected WS-Flex graphs

Optional (recommended, low marginal cost):
- ARM_SCORE: select top-B by a chosen proxy score (no TPE) to separate “proxy usefulness” from “TPE usefulness”
- ARM_ORACLE: select top-B by true y=-maxRD after training (upper bound / headroom estimate)

============================================================
MODULE M1 — METRIC/PROXY COMPUTATION SUITE
============================================================

M1.1 Core graph metrics computed from A (undirected)
Compute:
- n = H; m = |E|
- density = 2m / (n(n-1))
- degrees d_i = Σ_j A_ij; deg_mean, deg_std
- clustering:
  C_i = 2 t_i / (d_i(d_i-1)) for d_i≥2; C = mean_i C_i
- path length:
  L = mean_{i≠j} dist(i,j)
- small-worldness sigma:
  σ = (C/C_rand) / (L/L_rand), where baseline rand is matched on (n,m) or degree seq
- Laplacian spectrum:
  L = D-A; L_norm = I - D^{-1/2} A D^{-1/2}
  algebraic connectivity: a(G) = λ2(L)
  normalized gap: gap_Lnorm = λ2(L_norm)
- spectral measures:
  spectral radius ρ(A) = max_k |λ_k(A)|
  adjacency spectral gap: gap_A = λ1(A)-λ2(A); also gap_A / max(λ1,ε)
- effective resistance:
  compute pseudoinverse L^+; R_ij = L^+_ii + L^+_jj - 2L^+_ij
  Kirchhoff index: Kf = n * tr(L^+); avg resistance: Rbar = 2Kf/(n(n-1))
- centralities (compute node distributions; then summarize):
  betweenness BC(v), closeness CC(v), eigenvector centrality EC(v)
  summarize mean/std/max and Gini for each
- motifs:
  triangles T = tr(A^3)/6; normalize by C(n,3)
  optional 4-cycles C4 formula; normalize by C(n,4)

M1.2 Metrics computed from oriented adjacency Ã (directed)
Compute:
- directed spectral radius: ρ(Ã) = max_k |λ_k(Ã)|
- directed motif counts (optional): feed-forward loops, 3-node motifs
- in/out degree summaries and imbalance measures

M1.3 Entropy proxies
- Degree entropy:
  p_i = d_i/(2m); H_deg = -Σ_i p_i log p_i
  TE = H_deg / log(n)
- Spectral entropy proxy (optional, to match some literature):
  H_spec = log(ρ(A))

M1.4 ORC (Ollivier–Ricci curvature)
For each edge (x,y):
- define p_x(x)=α, p_x(z)=(1-α)/deg(x) for z∈N(x); α fixed at 0.5
- compute W1(p_x,p_y) using shortest-path distances as ground metric
- κ(x,y) = 1 - W1(p_x,p_y)   (since d(x,y)=1)
Aggregate:
- ORC_mean = mean_edges κ(x,y)
- ORC_min  = min_edges κ(x,y)

M1.5 Residualization and normalization
Because many metrics depend on k (degree regime), define within-k residuals:
For metric M:
- estimate μ_M(k) and σ_M(k) from a reference sample per k (>=200 graphs/k)
- M_res = M - μ_M(k)
- M_z   = (M - μ_M(k)) / (σ_M(k)+ε)
Use M_z for modeling/selection; use M_res for plots.
Clip M_z to [-5,5] for stability.

Outputs of Module M1:
- metrics.csv: one row per graph with all metrics, residuals, z-scores, and identifiers
- diagnostic plots: distributions per regime; corr(M,k)

GO/NO-GO checkpoint after M1:
- GO if metrics compute stably (no NaN explosion) and corr(M,k) is reduced after residualization (|corr| < 0.15 for key proxies TE_z, sigma_z, ORC_z).
- NO-GO only if metrics are unstable/uncomputable at scale; in that case, drop problematic metrics and proceed with stable subset.

============================================================
MODULE M2 — PROXY VIABILITY (LIGHTWEIGHT LABELED CHECK)
============================================================

Goal: Decide if any proxy (or composite) predicts robustness enough to justify heavy selection experiments.

M2.1 Sampling design
- Create N_pool = 64 WS-Flex graphs stratified by regime:
  16 graphs per regime, k sampled uniformly within regime list, p ~ Uniform(0,1)
- Compute all proxies via Module M1

M2.2 “Cheap labels” training for proxy validation
- Start with 1 subject (subject 1) for speed
- Training seeds: S_pilot = 1 (minimum); S_pilot = 2 preferred
- Evaluate robustness labels y = -maxRD at target_snr_db = -6 only (less saturation risk)
- Use full evaluation protocol otherwise (same alpha_grid)

M2.3 Proxy predictiveness tests
For each proxy P and label y:
- Pearson r and Spearman ρ; compute p-values
- Partial correlation controlling k (regress P and y on k; correlate residuals)
- Mutual information (KNN estimator) vs shuffled baseline
- AUC for top-25% robust classification
- Cross-validated surrogate models:
  - Ridge regression on z-features
  - Gradient-boosted trees on z-features
Report cross-validated Spearman(ŷ,y), R^2, and MAE

M2.4 Required diagnostics
- Scatter plots: P vs y, colored by regime
- Heatmap: mean y over (C,L) tertiles per regime
- Pareto: p_clean vs maxRD

M2.5 GO/NO-GO thresholds
PROXY VIABLE (GO to M3) if at least one is true:
- Single proxy: Spearman ρ ≥ 0.35 with p<0.05 (FDR corrected) AND AUC ≥ 0.70
- Composite surrogate: CV Spearman ≥ 0.45 AND MAE improves ≥15% over constant baseline
PLUS: sign consistency across ≥2/3 subjects in a small confirmatory check (optional quick pass with subjects {1,3,4} and S=1)

If PROXY NOT VIABLE:
- NO-GO to “TPE > random” claim
- Jump to pivot Module M5 (basin framing)

============================================================
MODULE M3 — SELECTION PROTOCOL ABLATION (NO HEAVY TRAINING YET)
============================================================

Goal: Ensure the selection protocol itself is not washing out differences.

M3.1 Define two selection constraints
- MODE_REGIME:
  select exactly 2 graphs per regime (B=8 total), no (C,L) bin requirements
- MODE_NONE:
  select any B=8 graphs (unconstrained)

M3.2 Arms
- ARM_RAND: random selection under each mode
- ARM_TPE: TPE selection using chosen proxy objective under each mode
Optional:
- ARM_SCORE: choose top-B by a proxy score (no TPE)

M3.3 Proxy budget
- Fix proxy evaluation budget M_budget (e.g., 512 graphs) for both arms
- Ensure both arms see the same generated graph stream / candidate set (same seeds)

Outputs:
- selected_architectures_MODE_REGIME.csv
- selected_architectures_MODE_NONE.csv
- selection diagnostics: where selected graphs land in proxy space and (C,L) space

GO/NO-GO checkpoint after M3:
- GO if TPE under MODE_NONE produces a distinctly shifted proxy distribution vs random (e.g., |Δ mean proxy_z| ≥ 0.5)
- If no shift even in proxy space, TPE is effectively not optimizing; debug objective/implementation before any heavy training.

============================================================
MODULE M4 — FINAL HEAVY TRAINING HEAD-TO-HEAD (DECISIVE)
============================================================

Entry requirement: Modules M2 and M3 must be GO.

M4.1 Training configuration
- subjects: {1,3,4}
- seeds: S_heavy = 2 minimum (S=3 preferred if runtime allows)
- B=8 per arm per mode (MODE_REGIME and MODE_NONE)
- Train once; evaluate at both target_snr_db = -12 and -6 (evaluation-only change)

M4.2 Metrics to report
Per graph:
- p_clean, maxRD, max_drop, AUPC
- seed variance estimates
Aggregate per arm:
- mean/median and 95% CI for maxRD (primary), plus secondary metrics
- Pareto front visualization (p_clean vs maxRD)

M4.3 Statistical inference
- Hierarchical bootstrap:
  preferred: subject → graph → seed
  fallback: graph → seed (subject fixed)
- Pairing:
  Since MODE_REGIME is stratified, compute regime-wise differences and aggregate (reduces confounding)
- Report effect sizes:
  Cohen’s d (graph-level) for maxRD and max_drop

M4.4 DECISIVE GO thresholds (claim “TPE > random” supported) — must meet ALL:
- Primary: Δ = E[maxRD_rand - maxRD_tpe] ≥ 0.05
- 95% bootstrap CI for Δ excludes 0
- Cohen’s d ≥ 0.5
- Directional consistency across ≥2/3 subjects at BOTH SNR settings OR at least at -6 with a clear explanation if -12 saturates

M4.5 DECISIVE NO-GO thresholds (trigger pivot) — if ANY is true:
- |Δ| < 0.02 with CI including 0 at both SNR settings
- Gains appear only by sacrificing clean ROC-AUC by >0.02 in MODE_REGIME
- TPE only wins in MODE_NONE and loses in MODE_REGIME (suggests regime reallocation rather than within-regime superiority), unless paper framing explicitly adopts “best-of-basin via reallocation” as the claim

============================================================
MODULE M5 — PIVOT PLAN (IF NO-GO)
============================================================

New Plot 2 thesis: “WS-Flex defines a solution-rich topology basin for robust CfC graphs under capacity control; sophisticated selection yields limited marginal gains.”

Required experiments/figures (low extra cost once heavy training is done):
- Basin density:
  heatmaps of robustness (y=-maxRD) over (C,L) bins per regime
- Family effect:
  compare WS-Flex basin (random/regime-stratified) vs external random wiring under matched capacity
- Practical selection:
  show that regime-stratified random + light proxy filtering matches TPE performance
- Headroom estimate:
  compute ORACLE top-B from trained pool to quantify remaining improvement “available” vs what TPE captures

GO criteria for pivot readiness:
- Demonstrate robust graphs exist at high frequency in WS-Flex (e.g., top quartile robustness contains ≥25% of graphs in multiple regimes)
- Demonstrate family effect vs external random is statistically significant (CI excludes 0 and meaningful effect size)

============================================================
RUNTIME / COMPUTE ESTIMATES (rough, user to fill with measured T_job)
============================================================

Let T_job = GPU time for one (graph, seed, subject) training+eval job.

- Module M2 pilot (subject=1):
  jobs ≈ N_pool * S_pilot * 1 = 64 * S_pilot
  time ≈ 64*S_pilot*T_job

- Module M4 heavy (two modes):
  jobs ≈ (modes=2) * (arms=2) * B * S_heavy * subjects
       = 2 * 2 * 8 * S_heavy * 3 = 96 * S_heavy
  time ≈ 96*S_heavy*T_job
  (evaluation at two SNRs is mostly extra inference, not retraining)

Stop-loss rule:
- If Module M2 fails, do NOT run M4. Pivot using M5.

END SPEC
