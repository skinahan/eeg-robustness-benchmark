PAPER 3 TOY ROBUSTNESS BENCHMARK (TASK-AGNOSTIC, TIME-SERIES, DYNAMICS-ALIGNED)
Goal: Evaluate whether proxy-guided TPE NAS (graph/topology search) yields measurably more robust recurrent dynamics than (i) random topology selection and (ii) random + proxy filtering, under a fast, domain-agnostic time-series task. Include CfC (searched + random), NCP, and LSTM under capacity-matched constraints.

=====================================================================
A) CORE DESIGN DECISIONS (WITH MOTIVATION)
=====================================================================
D1. Use a synthetic time-series task with:
    - fully controlled ground-truth generation (removes dataset confounds),
    - cheap training (many topologies evaluated quickly),
    - perturbations that mimic real-world corruption (noise, drift, impulses).
    Motivation: isolate architectural/topological contributions to robustness.

D2. Use metrics that assess BOTH:
    (i) task-level robustness (performance degradation vs perturbation intensity),
    (ii) dynamics-level robustness (hidden-state sensitivity/stability).
    Motivation: in continuous-time / dynamical models (CfC/LTC/NCP), robustness
    often appears first in state dynamics before accuracy differences emerge.

D3. Compare 3 topology selection regimes (for CfC-based searched class):
    R1: Proxy-guided TPE NAS (your method).
    R2: Random selection (uniform over graph space).
    R3: Random + proxy filtering (sample many; keep top-q by proxy).
    Motivation: directly test whether TPE adds value beyond metric screening.

D4. Capacity matching across model families:
    - Fix a parameter budget P_target and enforce |P_model - P_target| / P_target <= 5%.
    - Optionally also constrain forward FLOPs within ±10% (if easy to compute).
    Motivation: avoid robustness differences explained by raw capacity.

D5. Keep the input encoder identical across models (same preprocessing, same MLP/linear).
    Motivation: remove representational confounds and isolate recurrence/topology.

=====================================================================
B) TASK: NOISY HARMONIC OSCILLATOR CLASSIFICATION
=====================================================================
Task: binary classification of frequency regime (low vs high) given a length-T sequence.

1) Time grid and base signal
- Choose sequence length T (e.g., T=256) and timestep dt (e.g., dt=1.0; units arbitrary).
- Time indices: t_i = i * dt, i=0,...,T-1.

2) Clean signal generator
For each sample n:
- Sample amplitude: A ~ Uniform[A_min, A_max] (e.g., [0.5, 1.5])
- Sample phase:     phi ~ Uniform[0, 2π]
- Sample frequency:
    y=0 (low):  ω ~ Uniform[ω_L_min, ω_L_max]
    y=1 (high): ω ~ Uniform[ω_H_min, ω_H_max]
  Example: ω_L in [0.05, 0.15], ω_H in [0.20, 0.35] radians/step.
- Clean signal:
    x_clean(t_i) = A * sin(ω * t_i + phi)

3) Optional nuisance variation (cheap realism)
- Add mild amplitude modulation (optional):
    m(t_i) = 1 + a_m * sin(ω_m * t_i + phi_m), with small a_m (e.g., 0.1)
    x_clean(t_i) <- m(t_i) * x_clean(t_i)
Motivation: discourages trivial frequency heuristics; keeps task simple.

4) Input dimensionality
- Base: 1D signal (C=1 channel).
- Optional: augment to C>1 by adding correlated channels:
    x_c(t) = x_clean(t) + ρ_c * ε_c(t)  (small), or fixed linear mixing.
Motivation: tests multi-channel robustness without domain specificity.

5) Dataset sizes (suggested defaults)
- Train: N_train = 20k
- Val:   N_val   = 2k
- Test:  N_test  = 5k
Motivation: small enough to run quickly; large enough for stable curves.

6) Splits
- Stratified by class.
- Fix random seeds; keep same splits across all models/regimes.

=====================================================================
C) PERTURBATION SUITE (ROBUSTNESS STRESSORS)
=====================================================================
We define a perturbation operator P_k(x; α) that produces x' from clean x, parameterized by intensity α.
Evaluate a grid of α values per perturbation type.

Perturbation types:
P1) Additive Gaussian noise (AWGN)
- x'(t_i) = x(t_i) + η_i,  η_i ~ Normal(0, σ^2)
- Intensity grid: σ ∈ {0.0, 0.05, 0.10, 0.20, 0.30} * std(x_clean)
Motivation: simplest stochastic corruption; aligns with many robustness studies.

P2) Impulse / spike noise
- Choose K_imp ~ Poisson(λ) spikes; pick indices {i_k}; magnitudes u_k ~ Normal(0, s^2).
- x'(t_i) = x(t_i) + Σ_k u_k * 1[i = i_k]
- Intensity grid: λ ∈ {0, 1, 2, 4, 8} per sequence (or per 256 steps)
Motivation: mimics sensor glitches and transient artifacts (time-local outliers).

P3) Low-frequency drift (baseline wander)
- d(t_i) = B * sin(ω_d * t_i + φ_d), ω_d small (e.g., ω_d ∈ [0.005, 0.02])
- x'(t_i) = x(t_i) + d(t_i)
- Intensity grid: B ∈ {0.0, 0.1, 0.2, 0.3, 0.5} * std(x_clean)
Motivation: stresses long-timescale stability, relevant to continuous-time recurrence.

P4) Time-warp (optional stretch goal; can be omitted initially)
- Define a monotone warp w(i) = i + Δ(i), with Δ smooth and bounded.
- Resample x at warped indices (linear interp).
- Intensity grid: max|Δ| ∈ {0, 2, 4, 8, 16} steps
Motivation: tests invariance to temporal misalignment.

Evaluation protocol per perturbation:
- For each α in grid, generate perturbed test set x_test^(k,α).
- Use identical perturbation realizations across models for fairness:
  either (i) store perturbation seeds per sample, or (ii) precompute and cache perturbed arrays.

=====================================================================
D) MODELS AND CAPACITY MATCHING (CfC + NCP + LSTM)
=====================================================================
Common components (shared across all models):
- Input normalization: per-sample standardization (optional) or global standardization from train set.
- Input encoder E:
    e_t = Linear(C -> D_in) applied per timestep (or small 1-layer MLP).
  Keep E identical across models to isolate recurrence.
- Readout head:
    y_hat = Linear(H -> 2) on final state h_T (or pooled state mean; pick one and keep fixed).

D.1) CfC family (searched vs baselines)
- Recurrent core: CfCCell or CfC layer unrolled over t=1..T.
- Topology/wiring: the internal sparse connectivity pattern (graph) is what search varies.
- Regimes:
  R1 CfC-TPE: proxy-guided TPE NAS selects graphs/topologies.
  R2 CfC-Rand: random graph selection.
  R3 CfC-Rand+Filter: sample M random graphs, keep top-q by proxy, train/eval those.

Implementation adaptation opportunities:
- Reuse your existing graph generator + proxy computation + TPE loop.
- Replace EEG feature extractor with the common encoder E (above).
- Keep training harness identical to reduce implementation drift.

D.2) NCP baseline
- Use an NCP (sparse, structured recurrent circuit) with parameter count matched to P_target.
- If using an existing library: configure sensory/inter/neuron counts to match budget.
- If your codebase already supports NCP/CfC: reuse that pathway.
Motivation: directly aligned with Lechner/Hasani’s circuit policy framing; strong comparator.

D.3) LSTM baseline
- Single-layer LSTM (or 2-layer if needed for param match), hidden size H_LSTM tuned to match P_target.
- Ensure same encoder E and same readout head.
Motivation: strong conventional recurrent baseline; reviewers expect it.

D.4) Capacity matching procedure (required)
Step CM1: Choose a reference model to define P_target:
- Recommend: pick a mid-sized CfC baseline (e.g., CfC with hidden size H_ref and typical wiring).
- Compute P_target = number of trainable parameters.

Step CM2: For each model family, choose hyperparameters to satisfy:
- |P_model - P_target| / P_target <= 0.05 (5%).
- Record P_model for reporting.
Optional CM3 (nice-to-have): approximate FLOPs per forward pass and constrain within ±10%.

Notes:
- For LSTM params (approx):
  P_LSTM ≈ 4 * [H*(H + D_in) + H]  (weights + biases), plus encoder/readout params.
- For CfC/NCP, compute exact params from implementation (preferred).

=====================================================================
E) TRAINING PROTOCOL (FAST, FAIR, STABLE)
=====================================================================
Optimizer and schedule (suggested defaults):
- AdamW, lr=1e-3, weight_decay=1e-4
- Batch size: 256
- Epochs: 20 (early stopping on val loss with patience=3)
- Loss: cross-entropy
- Seeds: S = {0,1,2,3,4} (at least 5 seeds) for stability.

Training fairness requirements:
- Same dataset splits and augmentations across all models.
- Same stopping criteria.
- Same number of training steps (or early-stopping rule) per seed.

Important: No robustness-specific training at first (train only on clean data).
Motivation: evaluate inherent robustness of dynamics/topology, not trained defenses.
Optional extension later: train with a mild mixture of perturbations and compare deltas.

=====================================================================
F) METRICS (TASK-LEVEL + DYNAMICS-LEVEL)
=====================================================================
F.1 Task-level performance
Compute per perturbation type k and intensity α:
- Accuracy_k(α) on perturbed test set.
- Optionally ROC-AUC_k(α) if you output probabilities (recommended for smoother curves).

Define clean performance:
- Acc_clean = Accuracy(α=0)
- AUC_clean = ROC-AUC(α=0)

Define Relative Degradation (RD) at intensity α:
- RD_k(α) = (Score_clean - Score_k(α)) / max(ε, Score_clean)
  where Score is AUC or Acc; use ε=1e-6.

Define Max RD per perturbation:
- MaxRD_k = max_{α>0} RD_k(α)

Define AUPC-style robustness summary per perturbation:
- AUPC_k = (1 / |A|) * Σ_{α in grid} Score_k(α)
  (or trapezoidal integration over α; either is fine but keep consistent)

Motivation: mirrors your EEG robustness reporting (AUPC, RD) while staying domain-agnostic.

F.2 Dynamics-level robustness (critical for Paper 3)
Compute on the test set (or a fixed subset, e.g., 512 samples per class) to reduce compute.

Let h_t(x) be hidden state at time t for input sequence x.

M1) Empirical input sensitivity coefficient (approx Lipschitz)
For a given sample x:
- Create x_ε = x + ε * δ, where δ is a fixed-norm random direction:
  δ ~ Normal(0, I), then δ <- δ / ||δ||_2
- Measure final-state sensitivity:
  S(x; ε) = || h_T(x_ε) - h_T(x) ||_2 / || x_ε - x ||_2
Aggregate:
- Sensitivity(ε) = mean_x S(x; ε)
Choose ε ∈ {1e-3, 1e-2} * ||x||_2 (or set relative to std).
Motivation: cheap, model-agnostic proxy for stability to small perturbations.

M2) Hidden-state variance under stochastic noise
For each clean x:
- Generate R noise realizations {x^(r)} via AWGN with fixed σ (e.g., σ=0.2*std).
- Compute Var over realizations at each t:
  V_t(x) = Var_r [ h_t(x^(r)) ] (elementwise), then take trace or mean over dims.
Aggregate:
- StateVar = (1/T) * mean_x Σ_t mean_dim V_t(x)
Motivation: measures stability of internal trajectories under realistic noise.

M3) Empirical contraction / divergence rate (Lyapunov-style estimate)
For each x, create a nearby input x̃ = x + εδ as above.
Track distance over time:
- d_t = || h_t(x̃) - h_t(x) ||_2
Define contraction rate:
- λ(x) = (1/(T-1)) * Σ_{t=2..T} log( (d_t + ε0) / (d_{t-1} + ε0) )
with ε0 = 1e-8 to avoid log(0).
Aggregate:
- Lambda = mean_x λ(x)
Interpretation:
- More negative Lambda => stronger contraction (more stable dynamics).
Motivation: directly tied to continuous-time dynamical stability intuition.

Reporting:
- For each model/regime: report (Acc/AUC clean), (AUPC_k, MaxRD_k) for P1-P3,
  plus Sensitivity, StateVar, Lambda.

=====================================================================
G) NAS / TOPOLOGY SEARCH PROTOCOL (CfC ONLY)
=====================================================================
Define a fixed search space S of graphs/topologies (same as Paper 3 core).
- Example: WS-Flex parameterization (rewire prob p, degree regime, etc.), or your current space.

Proxy metric:
- Use your existing proxy score f_proxy(g) (graph -> scalar).
- If multiple proxy metrics exist, define a single combined proxy via weighted sum or Pareto rank.
  Keep weights fixed across all experiments.

Regimes:
R1 (TPE NAS):
- Budget: B_evals graphs (e.g., 50 or 100).
- TPE suggests next graph based on past (graph params -> proxy/utility).
- Select top K graphs by final objective (below) for training.

R2 (Random):
- Sample B_evals graphs uniformly from S.
- Select top K by final objective.

R3 (Random + Proxy Filter):
- Sample M >> K graphs (e.g., M=500).
- Keep top q% by proxy (e.g., q=10%), then randomly choose K from that subset OR choose top-K by proxy.
- Train/eval those K graphs.

Final objective for selecting “best graphs to train” (must be identical across regimes):
Option A (fastest): select graphs to train solely by proxy (then evaluate trained results).
Option B (more faithful): train a small number of steps (few epochs) as a cheap “inner loop” score.
Given your compute constraints and Paper 3 design, recommend Option A for this toy, to stay aligned.

K (trained graphs per regime):
- Suggest K=10 (per regime) for robust statistics without huge compute.

=====================================================================
H) ANALYSIS AND STATISTICS
=====================================================================
Per model family and regime:
- Run across S seeds (>=5) and compute mean ± 95% CI for:
  clean score, AUPC_k, MaxRD_k, Sensitivity, StateVar, Lambda.

Primary comparisons (what you want to see):
C1: CfC-TPE vs CfC-Rand on robustness summaries (AUPC, MaxRD) and dynamics metrics.
C2: CfC-TPE vs CfC-Rand+Filter to isolate “TPE benefit beyond proxy screening.”
C3: Best CfC-TPE vs NCP vs LSTM to show relevance beyond CfC space.

Expected patterns if topology search genuinely improves robustness:
- Similar clean score across CfC regimes (not guaranteed, but acceptable),
- Lower Sensitivity, lower StateVar, more negative Lambda for CfC-TPE,
- Flatter degradation curves (higher AUPC, lower MaxRD) for CfC-TPE.

=====================================================================
I) IMPLEMENTATION CHECKLIST (REUSE VS NEW WORK)
=====================================================================
Reuse / adapt from existing Paper 3 code:
- Graph generator + proxy metric computation.
- TPE loop and random sampling baseline.
- Training harness (multi-seed training, logging, aggregation utilities).

New (minimal) functionality to implement:
- Synthetic dataset generator for harmonic oscillators.
- Perturbation suite P1-P3 (and caching).
- Dynamics metrics M1-M3 (Sensitivity, StateVar, Lambda) computed from hidden states.
  NOTE: requires access to per-timestep hidden states h_t for each model.
  If your forward pass currently returns only final logits, add an option:
    forward(..., return_states=True) -> {logits, states[T,H]}

Nice-to-have (optional):
- FLOPs estimation for tighter capacity matching.
- Time-warp perturbation P4.

=====================================================================
J) MINIMAL DIAGNOSTIC / TEST RUNS (DO THESE FIRST) + GO/NO-GO
=====================================================================

TEST 1: Data sanity + perturbation sanity (NO TRAINING)
- Generate 1000 samples; verify class balance.
- Plot/inspect a few sequences for each perturbation at min/max α (manual check).
Go/No-Go:
- GO if perturbations visibly match intended behavior (Gaussian looks noisy; impulse has spikes; drift is slow).
- NO-GO if perturbations change label semantics (e.g., drift dominates signal completely at moderate α); adjust intensity grids.

TEST 2: Single-model overfit check (FAST)
- Train one small model (e.g., LSTM) on N_train=2000 for 30 epochs.
- Expect near-100% train accuracy and strong val accuracy (>90%) on clean.
Go/No-Go:
- GO if model can learn task quickly (val AUC/Acc > 0.90).
- NO-GO if learning stalls (<0.75): widen frequency separation, reduce nuisance variation, or increase T.

TEST 3: Capacity matching validation (STATIC)
- Instantiate CfC, NCP, LSTM and compute param counts.
Go/No-Go:
- GO if all within 5% of P_target.
- NO-GO if mismatch: adjust H_LSTM / NCP neuron counts / CfC hidden size.

TEST 4: Hidden-state extraction correctness (STATIC FORWARD)
- For each model type, run forward with return_states=True on a batch.
- Verify tensor shapes: states [B, T, H] (or [T,B,H] consistently).
Go/No-Go:
- GO if states are returned and finite (no NaNs/Infs).
- NO-GO if not: fix instrumentation before any NAS runs.

TEST 5: Dynamics metrics smoke test (NO TRAINING OR MINIMAL)
- On an untrained model or minimally trained (1 epoch), compute M1-M3 on 128 samples.
- Ensure metrics are finite and stable across runs.
Go/No-Go:
- GO if metrics are finite and vary sensibly with ε and σ (Sensitivity increases with ε; StateVar increases with σ).
- NO-GO if metrics are noisy/degenerate: increase sample count or adjust ε, ε0.

TEST 6: Mini separation pilot (THE FIRST REAL SIGNAL)
- Regimes: CfC-TPE (B_evals=20, K=3), CfC-Rand (B_evals=20, K=3), CfC-Rand+Filter (M=100, q=10%, K=3)
- Seeds: 2
- Train: 10 epochs
- Evaluate: clean + AWGN only (P1) across σ grid, plus M1 and Lambda only.
Go/No-Go:
- GO if CfC-TPE shows at least ONE of:
    (i) higher AUPC_P1 by >= 0.02 absolute, OR
    (ii) lower Sensitivity by >= 10% relative, OR
    (iii) more negative Lambda by >= 0.05 (absolute) compared to both baselines,
  consistently across 2 seeds (directionally consistent).
- NO-GO if no consistent directional advantage: revisit proxy definition, search space, or task difficulty (e.g., increase drift/impulse stressors; or reduce clean difficulty so robustness dominates).

If TEST 6 is a GO:
- Scale to full protocol (K=10, seeds=5, P1-P3) and add NCP/LSTM baselines.

END OF SPEC.
