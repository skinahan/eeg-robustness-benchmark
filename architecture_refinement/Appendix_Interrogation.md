Below is a **section-by-section checklist of concrete questions** that must be answered to fully flesh out the Appendix into a rigorous, reproducible technical supplement.

---

# A. Full Method Details

---

## 1. Graph Generator Families and Parameterization

### A.1 Define θ completely

We need to specify:

1. **What exactly is θ?**

   * Is θ = (H, k, p, seed)?
   * Are there additional hidden parameters (e.g., orientation seed, sparsity cap, rewiring policy)?
   * Are k and p continuous or discretized?
   * Are regimes encoded explicitly in θ or derived from (k, p)?

2. **Node count**

   * Is H always fixed (e.g., H=32)?
   * Does BO ever vary H?
   * If not, state explicitly.

3. **k (degree parameter)**

   * Is k even-only?
   * What are allowed values?
   * Is k defined as initial lattice degree in WS?
   * Is mean degree exactly preserved after rewiring?

4. **p (rewiring probability)**

   * Continuous [0,1]?
   * Discretized grid?
   * Uniform sampling or log-scale?
   * Is p clipped or rounded?

5. **WS-Flex specifics**

   * How does WS-Flex differ from canonical Watts–Strogatz?
   * Is degree strictly homogeneous?
   * Any multi-edges allowed?
   * Self-loops allowed?
   * Is rewiring per edge or per node?

6. **Pruning rules**

   * Do we discard disconnected graphs?
   * Do we reject graphs with isolated nodes?
   * Do we reject extreme clustering/path length regimes?
   * Is there a cap on max degree?
   * Is there a minimum edge count?

7. **Regime definitions**

   * How are “near-regular”, “small-world”, “near-random” defined?
   * Are regimes based on p thresholds or derived graph metrics?
   * What are exact numeric boundaries?

---

## 2. Graph Metrics: Definitions and Computation

For each metric:

### A.2 Implementation-level questions

1. How exactly is clustering coefficient computed?

   * Global vs average local?
   * Weighted or unweighted?
   * Directed or undirected?

2. How is average shortest path length computed?

   * Only for connected graphs?
   * Infinite paths handled how?
   * Which algorithm (Floyd–Warshall? Dijkstra? NetworkX default?)?

3. Computational complexity:

   * O(H³)? O(H log H)?
   * Does this scale safely for H=32?

4. Are metrics computed on:

   * Undirected graph?
   * Directed graph after orientation?

5. How are ER reference values computed?

   * Closed-form approximations?
   * Empirical sampling?
   * How many samples?
   * Matched exactly on edge count or density?

6. Is curvature (if used) computed?

   * Which curvature? Ollivier-Ricci?
   * Using which library?
   * Is it approximated?

7. Determinism:

   * Are metric values deterministic under fixed seed?
   * Any floating point nondeterminism?

---

## 3. Constrained Multi-Objective BO Details

### A.3 Surrogate modeling

1. What surrogate model?

   * TPE (Optuna)?
   * Gaussian Process?
   * Independent per objective?

2. Multi-objective handling:

   * Weighted sum?
   * True Pareto TPE?
   * Separate scalarizations?

3. Acquisition function:

   * Expected improvement?
   * Hypervolume improvement?
   * Custom?

4. Initialization:

   * Random trials count?
   * Latin hypercube?
   * Fixed seed?

5. Stopping:

   * Fixed trial budget?
   * Early stopping if convergence?
   * Hypervolume plateau?

6. Constraint modeling:

   * Hard filtering before surrogate?
   * Penalized objective?
   * Feasibility classifier?

7. Selection rule:

   * Final solution chosen how?
   * Pareto + knee?
   * Hypervolume max contributor?
   * Best scalarization under equal weights?

8. BO hyperparameters:

   * n_trials?
   * n_startup_trials?
   * gamma?
   * n_ei_candidates?

---

## 4. Graph-to-CfC Instantiation Algorithm

### A.4 Mapping adjacency to recurrence

1. Given adjacency matrix A:

   * How is it converted to hidden-to-hidden mask?
   * Is A binary?
   * Are weights masked post-initialization or structurally omitted?

2. Directed conversion:

   * How are undirected edges oriented?
   * Deterministic policy?
   * Seed-controlled?

3. Are self-loops added?

   * Always?
   * Required for CfC stability?

4. CfC specifics:

   * Does masking apply to:

     * All recurrent matrices?
     * Only W_h?
     * Gating matrices too?

5. Are input-to-hidden weights dense?

   * Always dense?
   * Masked?

6. Fixed components:

   * Readout head dense?
   * Biases unaffected?
   * LayerNorm present?

7. Hidden state dimension:

   * H matches node count?
   * One neuron per node?

---

# B. Experimental Details for Reproducibility

---

## 5. Datasets and Preprocessing

For each dataset:

1. Name and version?
2. License?
3. Number of subjects?
4. Sampling rate?
5. Channels used?
6. Preprocessing:

   * Bandpass filter?
   * Notch filter?
   * Referencing?
   * Baseline correction?
   * Artifact rejection?
7. Epoch window:

   * Time interval?
   * Aligned to what event?
8. Resampling?
9. Normalization:

   * Per subject?
   * Per session?
   * Z-score?
10. Splits:

    * Cross-session?
    * Cross-subject?
    * Exact fold construction?
11. Leakage prevention:

    * Subject isolation?
    * Session isolation?

---

## 6. Perturbations and Robustness Metrics

### A.5 Perturbation definitions

1. Gaussian noise:

   * Mean?
   * Covariance?
   * Channel-independent?
   * Based on training covariance?
   * Intensity defined as SNR or variance scalar?

2. Channel dropout:

   * Random channels?
   * Same channels per sample?
   * Fixed mask per trial?
   * Percentage grid?

3. Ocular artifact simulation:

   * Synthetic waveform?
   * Real EOG data?
   * Injected into frontal channels only?
   * Scaling grid?

4. Intensity grids:

   * Exact numeric grid for each perturbation?
   * Uniform spacing?
   * Log spacing?

---

### A.6 Robustness metrics

1. AUPC:

   * Area under what curve?
   * Accuracy vs intensity?
   * ROC-AUC vs intensity?
   * Integration method?

2. RD (robustness drop):

   * Max drop relative to clean?
   * Mean drop?
   * Worst-case?

3. Aggregation:

   * Per subject first?
   * Then average?
   * Weighted?

4. Error bars:

   * Std?
   * SEM?
   * 95% CI?

---

## 7. Training Protocol

1. Optimizer:

   * Adam?
   * AdamW?
   * Learning rate?
2. LR schedule:

   * Step?
   * Cosine?
   * None?
3. Batch size?
4. Epochs?
5. Early stopping:

   * Patience?
   * Metric monitored?
6. Weight decay?
7. Gradient clipping?
8. Mixed precision?
9. Determinism flags?
10. Number of seeds?
11. Random seed handling?
12. Validation selection?
13. Final model chosen:

* Best validation?
* Last epoch?

---

## 8. Capacity Accounting

1. Exact parameter count formula for CfC:

   * Hidden-to-hidden weights masked?
   * Input weights?
   * Biases?
   * Gating layers?
2. Does masking reduce parameter count or only zero weights?
3. How are FLOPs estimated?

   * Analytical?
   * Measured?
4. Is runtime dependent on sparsity?
5. Are models exactly matched on:

   * Total parameters?
   * Only recurrent parameters?

---

## 9. Compute Resources

1. GPU model?
2. CUDA version?
3. PyTorch version?
4. Average runtime per model?
5. Total search runtime?
6. Storage used?
7. Parallel trials?

---

# C. Extended Results

---

## 10. Per-dataset / per-perturbation breakdowns

1. Do we report:

   * Per-subject tables?
   * Only aggregated?
2. Separate by:

   * Gaussian?
   * Dropout?
   * EOG?
3. Statistical tests:

   * Paired?
   * Wilcoxon?
   * Corrected for multiple comparisons?

---

## 11. Metric–Robustness Analyses

1. Which correlations:

   * Pearson?
   * Spearman?
2. Per dataset or pooled?
3. Are correlations significant?
4. Robust to:

   * Removing extreme graphs?
   * Removing one regime?

---

## 12. Full Search Traces

1. How many trials?
2. Hypervolume progression?
3. Is hypervolume computed in normalized objective space?
4. Are infeasible trials shown?
5. Is best-so-far tracked?

---

# D. Ablations

---

## 13. Constraint Ablations

1. What happens if:

   * Connectivity constraint removed?
   * Sparsity band widened?
   * Budget increased?
2. How many valid graphs result?
3. Does BO efficiency change?
4. Final robustness impact?

---

## 14. Metric Set Ablations

1. If TE_res removed:

   * Does small-worldness dominate?
2. If σ removed:

   * Does entropy dominate?
3. Does Pareto collapse?
4. Does performance variance increase?

---

## 15. Generator Family Ablations

1. What alternative generators:

   * ER?
   * Random sparse?
2. Are they matched on degree?
3. Is BO rerun or fixed grid?
4. Do metrics behave differently?

---

# E. Metrics and Normalization

---

## 16. TE_res normalization specifics

1. Reference set size?
2. How is reference set sampled?
3. Stratified by k?
4. Fixed across experiments?
5. μ_TE(k) estimated how?
6. Are bounds clipped?
7. Are objectives:

   * Min-max normalized?
   * Z-scored?
   * Per trial dynamic?
8. Is normalization recomputed online?

---

# F. Feasibility Constraints and BO Integration

---

## 17. Infeasible trial handling

1. Are infeasible trials:

   * Pruned immediately?
   * Logged with dummy value?
2. Are they fed into surrogate?
3. Is feasibility modeled explicitly?
4. Does BO sample infeasible space repeatedly?
5. Is rejection rate reported?

---

# G. Pareto Selection and Coverage

---

## 18. Pareto extraction

1. Global pool or per regime?
2. Dominance defined strictly?
3. Floating tolerance?
4. Ties allowed?

---

## 19. Coverage-aware selection

1. Exact regime quotas?
2. Exact bin quotas?
3. Collapse cap definition?
4. How are bins defined?

   * Tertiles over fixed reference?
5. Deterministic ordering?

---

## 20. Edge Orientation Policy

1. Deterministic?
2. Seeded?
3. Based on node index?
4. Balanced in-degree/out-degree?

---

## 21. CfC Masking Location

We must specify equation-level placement:

* Does mask multiply W_h?
* Is mask applied before or after gating nonlinearity?
* Is mask binary or learnable?
* Is mask static during training?

---

## 22. Readout

1. Final hidden state?
2. Temporal mean pooling?
3. Logits per timestep?
4. Softmax applied where?

---

# I. Capacity Matching Section

---

## 23. Parameter-count equations

1. CfC full formula?
2. Masked recurrent matrix size?
3. Input projection?
4. Output head?
5. NCP partition sizes?

---

## 24. Capacity table

We need exact numeric counts at H=32 for:

* Dense CfC
* Random Sparse
* WS-Flex selected
* Random WS-Flex
* NCP

---

## 25. Runtime

1. Wall-clock measured?
2. Controlled?
3. Does sparsity reduce runtime?
4. Reported or not optimized?

---

# J. Perturbations and Evaluation

---

## 26. Evaluation protocol

1. Noise applied:

   * Test only?
   * Also training?
2. Calibration:

   * Per subject?
   * Global?
3. Clean retraining per perturbation?
4. Same model reused?

---

# K. Seeds and Statistics

---

## 27. Seeds

1. How many?
2. Fixed across models?
3. Averaging method?
4. Confidence intervals?

---

# L. Non-EEG Validation

---

## 28. Non-EEG tasks

1. What dataset?
2. Why chosen?
3. What perturbation?
4. Same BO used?
5. Same H?
6. Same metrics?
7. Supports generality claim how?

---

# M. Global Consistency Checks

We must verify:

1. No duplicated sections (instantiation appears twice).
2. All hyperparameters appear exactly once.
3. All normalization references match equations.
4. Feasibility definition consistent with BO code.
5. Degree regime definitions consistent everywhere.

---

# Final Note

To fully solidify the Appendix, we must be able to answer:

* Exactly how is every number in every plot computed?
* Can an external lab reimplement search + training + evaluation from this Appendix alone?
* Are all stochastic elements explicitly controlled?
