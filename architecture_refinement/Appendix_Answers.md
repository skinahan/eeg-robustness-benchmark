# Appendix Answers

This document answers every query in Appendix_Interrogation.md, with code citations for reproducibility.

---

# A. Full Method Details

---

## 1. Graph Generator Families and Parameterization

### A.1 Define θ completely

**1. What exactly is θ?**

- **Plain WS-Flex:** θ = (H, k, p, graph_seed). See `WSFlexParams` in [ws_flex_generator.py](architecture_refinement/ws_flex_generator.py) lines 25-37.
- **Modular WS-Flex:** θ additionally includes M, k_out, p_out, r_out, and seed_mod_params (for sampling M, p_out, r_out when not provided). See [ws_flex_generator.py](architecture_refinement/ws_flex_generator.py) lines 239-248.
- **Hidden parameters:** Orientation seed is derived deterministically: `wiring_seed = hash(graph_hash) mod 2^31-1` ([run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 320-322). Capacity filter (E_active regime bands) is optional via `--capacity_filter`.
- **k and p:** k is discretized via `suggest_categorical` over degree_regimes; p is continuous [0,1] via `suggest_float` ([run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 957-958).
- **Regimes:** Derived from k; not encoded in θ. Regimes are super_sparse, sparse, moderate, near_dense with k-value lists.

**2. Node count**

- H is fixed at 32. See [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) line 2375: `--H`, default 32.
- BO does not vary H; it is held constant. [NAS_plot2_topology_study_spec.txt](architecture_refinement/NAS_plot2_topology_study_spec.txt) line 41: "H = 32 (fixed across all conditions)."

**3. k (degree parameter)**

- **Even-only:** Yes. [ws_flex_generator.py](architecture_refinement/ws_flex_generator.py) lines 53-55: `if k % 2 != 0: k = k - 1 if k > 2 else 2`.
- **Allowed values:** 2 ≤ k ≤ H−1 (clamped in build_plain_ws_flex). Default degree_regimes: super_sparse {2,4,6}, sparse {8,10,12}, moderate {14,16,18}, near_dense {20,22,24,26} ([run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 2538-2541).
- **Definition:** k is the number of neighbors per node in the initial ring lattice (Watts-Strogatz). [ws_flex_generator.py](architecture_refinement/ws_flex_generator.py) line 47: "Neighbors per node in initial ring."
- **Mean degree after rewiring:** NetworkX `watts_strogatz_graph` preserves mean degree; rewiring swaps endpoints, so total edges unchanged.

**4. p (rewiring probability)**

- **Range:** Continuous [0, 1]. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) line 958: `trial.suggest_float("p", 0.0, 1.0)`.
- **Sampling:** Uniform over [0,1] (Optuna TPE default).
- **Clipping/rounding:** Not clipped or rounded; passed as float to `nx.watts_strogatz_graph`.

**5. WS-Flex specifics**

- **vs canonical Watts-Strogatz:** WS-Flex uses `nx.watts_strogatz_graph` plus `_ensure_connected` ([ws_flex_generator.py](architecture_refinement/ws_flex_generator.py) lines 56-57). If disconnected, bridge edges are added ([ws_flex_generator.py](architecture_refinement/ws_flex_generator.py) lines 148-162).
- **Degree homogeneity:** After rewiring, degree can vary slightly; WS preserves mean degree but not strict homogeneity.
- **Multi-edges:** No; NetworkX Graph does not allow multi-edges.
- **Self-loops:** No; `nx.watts_strogatz_graph` produces simple graphs.
- **Rewiring:** Per edge (standard WS algorithm).

**6. Pruning rules**

- **Disconnected graphs:** Rejected via `optuna.TrialPruned()` ([run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) line 965). Note: `_ensure_connected` repairs disconnection in ws_flex_generator, but `_make_ws_graph` (used in topology study) does not call it—disconnected graphs from `nx.watts_strogatz_graph` are pruned.
- **Isolated nodes:** Rejected implicitly (disconnected implies not all reachable).
- **Extreme clustering/path length:** No explicit rejection; capacity_filter optionally rejects by E_active regime band.
- **Cap on max degree:** No; k_max is H−1.
- **Minimum edge count:** Implicit via connectivity; no explicit minimum.

**7. Regime definitions**

- **Definitions:** Regimes are k-based: super_sparse (k ∈ {2,4,6}), sparse (k ∈ {8,10,12}), moderate (k ∈ {14,16,18}), near_dense (k ∈ {20,22,24,26}). [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 2538-2541.
- **Derivation:** From k only; not from p or graph metrics.
- **Numeric boundaries:** k boundaries per regime as above. Capacity bands: super_sparse E_active [32,96], sparse [112,192], moderate [208,288], near_dense [304,416] for H=32 ([capacity_utils.py](architecture_refinement/capacity_utils.py) lines 17-22, [plot2_bounds_v2.yaml](architecture_refinement/plot2_bounds_v2.yaml)).

---

## 2. Graph Metrics: Definitions and Computation

### A.2 Implementation-level questions

**1. Clustering coefficient**

- **Type:** Average local clustering. [small_world_metrics.py](architecture_refinement/small_world_metrics.py) line 34: `nx.average_clustering(G)`; [topology_analyzer.py](architecture_refinement/topology_analyzer.py) line 295.
- **Weighted:** Unweighted (default NetworkX behavior).
- **Directed/undirected:** Undirected; metrics computed on undirected graph before orientation.

**2. Average shortest path length**

- **Connected graphs:** Yes; for disconnected graphs, computed on largest component. [small_world_metrics.py](architecture_refinement/small_world_metrics.py) lines 36-43; [topology_analyzer.py](architecture_refinement/topology_analyzer.py) lines 298-306.
- **Infinite paths:** Disconnected pairs excluded; uses largest connected component.
- **Algorithm:** NetworkX default (BFS for unweighted).

**3. Computational complexity**

- Clustering: O(n) for average local. Path length: O(n²) for all-pairs. For H=32, both are negligible.

**4. Metrics computed on**

- Undirected graph, before orientation. [graph_metrics_suite.py](architecture_refinement/graph_metrics_suite.py), [small_world_metrics.py](architecture_refinement/small_world_metrics.py).

**5. ER reference values**

- **Method:** Monte Carlo by default. R_ER=20 connected ER graphs matched on p_edge. [small_world_metrics.py](architecture_refinement/small_world_metrics.py) lines 89-123, R_ER_DEFAULT=20.
- **Matching:** Matched on p_edge = 2|E|/(H(H−1)); edge count/density matched.
- **Analytic fallback:** `_er_analytic` used when Monte Carlo fails ([small_world_metrics.py](architecture_refinement/small_world_metrics.py) lines 77-86).
- **Cache:** (p_edge_bin, H) → (C_ER_mean, L_ER_mean) ([small_world_metrics.py](architecture_refinement/small_world_metrics.py) lines 26-27, 145-155).

**6. Curvature**

- **Type:** Ollivier-Ricci. [topology_analyzer.py](architecture_refinement/topology_analyzer.py) lines 172-179; [metrics_te_orc.py](architecture_refinement/metrics_te_orc.py) `ollivier_ricci_mean`.
- **Library:** Custom implementation using POT (if installed) or SciPy/HiGHS for Wasserstein-1.
- **Parameters:** alpha=0.5, max_edges=200 (subsampling for speed).
- **Approximation:** Edge curvature sampled over up to max_edges edges.

**7. Determinism**

- With fixed seed, graph generation is deterministic. Metric computation (NetworkX, NumPy) is deterministic. Floating-point operations are deterministic on same hardware.

---

## 3. Constrained Multi-Objective BO Details

### A.3 Surrogate modeling

**1. Surrogate model:** TPE (Optuna TPESampler). [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) line 944: `optuna.samplers.TPESampler(seed=int(base_seed), multivariate=True)`.

**2. Multi-objective:** True Pareto; `directions=["maximize","maximize"]`. Objectives: (TE_res, sigma) or (TE, ORC). [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 946-951.

**3. Acquisition:** TPE default (EI-like for multi-objective).

**4. Initialization:** Random trials; no Latin hypercube. TPE uses n_startup_trials (Optuna default).

**5. Stopping:** (a) M_max budget; (b) hv_window_saturation: relative HV improvement < 0.02 for 2 consecutive windows (W=3); (c) pareto_growth_saturation: < 5 new Pareto points for 3 consecutive batches. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 1105-1110; [NAS_plot2_topology_study_spec.txt](architecture_refinement/NAS_plot2_topology_study_spec.txt) lines 127-139.

**6. Constraint modeling:** Hard filter. Disconnected → `TrialPruned`. Optional capacity_filter → `TrialPruned` if E_active outside regime band. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 964-973.

**7. Selection rule:** Coverage-aware: regime_cl_bins_fixed with C/L tertiles; Pareto within bins; at least 2 per regime. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) `_select_topologies_coverage_aware`, lines 1574-1717.

**8. BO hyperparameters:** M0=200 (default), dM=100, M_max=1000; NAS spec lists M0=2000, dM=1000, M_max=10000. hv_window_W=3, hv_window_eps=0.02, hv_window_patience=2, pareto_new_m=5, pareto_patience_batches=3. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 2507-2522.

---

## 4. Graph-to-CfC Instantiation Algorithm

### A.4 Mapping adjacency to recurrence

**1. Adjacency → mask**

- Conversion: `_hidden_block_from_nx` in [arbitrary_wiring.py](architecture_refinement/arbitrary_wiring.py) lines 262-283 builds H×H matrix from graph edges.
- A is binary (1.0 for edges, 0 otherwise).
- Weights: Mask multiplies dense weights; masked positions are zeroed (structurally omitted in forward). [ncps-master/ncps/torch/cfc_cell.py](ncps-master/ncps/torch/cfc_cell.py) lines 139-140: `F.linear(x, self.ff1.weight * self.sparsity_mask, ...)`.

**2. Directed conversion**

- Undirected edges oriented via `hidden_edge_orientation="random_oriented"`: one direction per edge, `rng.integers(0,2)` per (i,j). [arbitrary_wiring.py](architecture_refinement/arbitrary_wiring.py) lines 286-300.
- Deterministic: seed-controlled (`self.seed`).
- Orientation seed: `s_orient(g) = hash(graph_hash) mod 2^31-1` ([run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 320-322).

**3. Self-loops**

- `add_hidden_self_loops=True` when hidden block would be empty; not added per edge. [arbitrary_wiring.py](architecture_refinement/arbitrary_wiring.py) line 94.

**4. CfC specifics**

- Mask applies to ff1, ff2, time_a, time_b (all Linear layers receiving concatenated input). [ncps-master/ncps/torch/cfc_cell.py](ncps-master/ncps/torch/cfc_cell.py) lines 139-155.
- Mask multiplies weight before linear: `weight * self.sparsity_mask`.
- Mask is static (`requires_grad=False`). [ncps-master/ncps/torch/cfc_cell.py](ncps-master/ncps/torch/cfc_cell.py) lines 73-79.

**5. Input-to-hidden**

- Dense by default (`input_strategy="dense"`). [arbitrary_wiring.py](architecture_refinement/arbitrary_wiring.py) lines 139-140: `W[0:I, I:I+H] = 1.0`.

**6. Fixed components**

- Readout: dense (`output_strategy="dense"`).
- Biases: Unaffected by mask (mask applied to weights only).
- LayerNorm: Not present in CfC cell.

**7. Hidden state dimension**

- H matches node count; one neuron per node. [arbitrary_wiring.py](architecture_refinement/arbitrary_wiring.py) `_hidden_size()`.

---

# B. Experimental Details for Reproducibility

---

## 5. Datasets and Preprocessing

**Per dataset (BNCI2014_001, Lee2019_SSVEP, BI2015a):**

- **Name/version:** MOABB datasets: BNCI2014_001, Lee2019_SSVEP, BI2015a. [experiment_config.yaml](experiment_config.yaml); [config.py](config.py) `get_paradigm`.
- **License:** MOABB/upstream dataset licenses apply.
- **Subjects:** BNCI2014_001: 9; Lee2019_SSVEP: 54; BI2015a: 43. [experiment_config.yaml](experiment_config.yaml).
- **Sampling rate:** BNCI2014_001: 250 Hz; Lee2019_SSVEP: 1000 Hz; BI2015a: 512 Hz. [config.py](config.py) `get_dataset_sampling_rate` lines 539-545.
- **Channels:** Dataset-dependent (MOABB).
- **Preprocessing:** Paradigm-defined. MotorImagery: fmin=8, fmax=35; SSVEP: tmin=0, tmax=4; P300: fmin=1, fmax=24, tmin=0, tmax=1. [config.py](config.py) lines 562-586; [BI2015A_P300_CONFIG.md](BI2015A_P300_CONFIG.md).
- **Epoch window:** Paradigm tmin/tmax. MotorImagery: tmin=0, tmax=None; SSVEP: 0–4 s; P300: 0–1 s.
- **Resampling:** Via paradigm `resample` parameter.
- **Normalization:** Per MOABB/paradigm (typically z-score or similar).
- **Splits:** WithinSession, CrossSession, CrossSubject via MOABB evaluators.
- **Leakage prevention:** Subject/session isolation via MOABB GroupKFold, LeaveOneGroupOut, etc.

---

## 6. Perturbations and Robustness Metrics

### A.5 Perturbation definitions

**1. Gaussian noise**

- Mean: 0. Covariance: i.i.d. (channel-independent).
- Scaling: `noise_scale = 4.0 * signal_rms * (intensity/100)`. [augmentation/noise.py](augmentation/noise.py) lines 839-845.
- Intensity: Percentage (10 = 10% noise relative to signal RMS).
- Channels: n_contam = int(n_channels * intensity/100), max(1,n_contam) if intensity>0; different channels per epoch.

**2. Channel dropout**

- Random channels; different per epoch. [augmentation/noise.py](augmentation/noise.py) lines 796-806.
- n_drop = int(n_channels * intensity/100), max(1,n_drop) if intensity>0.

**3. Ocular artifact (EOG)**

- Learned generic EOG mixing template. [augmentation/noise.py](augmentation/noise.py) lines 832-883.
- Intensity: temporal coverage (10% = 10% of time with artifacts).
- Injected via montage (frontal channels); `_generate_channel_names` for 22/32/62 channels.

**4. Intensity grids**

- 20 steps from min to saturation, uniform spacing via `np.linspace(min_intensity, max_intensity, num_steps)`. [utils.py](utils.py) `get_noise_intensities` lines 181-195; [experiment_config.yaml](experiment_config.yaml) num_steps: 20.
- Bounds from saturation_points_summary.csv; default (1.0, 50.0) if missing.

### A.6 Robustness metrics

**1. AUPC**

- Area under performance curve: mean of p(α) over intensity grid. [architecture_refinement/analyze_plot2_results.py](architecture_refinement/analyze_plot2_results.py); [PLOT 2 Final Overhaul Experiment Spec.md](architecture_refinement/PLOT%202%20Final%20Overhaul%20Experiment%20Spec.md) line 46: AUPC(g) = mean_{α∈A} p_g(α).
- Integration: discrete mean over alpha grid.

**2. RD (robustness drop)**

- RD_g(α) = (p_clean - p_g(α)) / max(p_clean - 0.5, ε), ε=1e-3. [analyze_plot2_results.py](architecture_refinement/analyze_plot2_results.py) lines 280-331; [NAS_pilot_study_spec.txt](architecture_refinement/NAS_pilot_study_spec.txt) lines 182-184.
- maxRD = max over α. Lower maxRD = more robust.

**3. Aggregation**

- Per seed first, then per graph/model; hierarchical bootstrap for CIs. [analyze_plot2_results.py](architecture_refinement/analyze_plot2_results.py).

**4. Error bars**

- 95% CI via hierarchical bootstrap. [analyze_plot2_results.py](architecture_refinement/analyze_plot2_results.py).

---

## 7. Training Protocol

- **Optimizer:** Adam/AdamW (model-dependent). [evaluation/two_stage_hp_opt.py](evaluation/two_stage_hp_opt.py); [NAS_plot2_topology_study_spec.txt](architecture_refinement/NAS_plot2_topology_study_spec.txt): weight decay disabled for masked recurrent weights.
- **LR schedule:** Model-dependent; often fixed or cosine.
- **Batch size:** Model-dependent.
- **Epochs:** DEFAULT_MAX_EPOCHS=200. [globals.py](globals.py) line 76.
- **Early stopping:** Patience=20, monitor=valid_loss, threshold=1e-5. [globals.py](globals.py) lines 55-72.
- **Weight decay:** Standard except masked recurrent (weight_decay=0). [NAS_plot2_topology_study_spec.txt](architecture_refinement/NAS_plot2_topology_study_spec.txt) lines 60-63.
- **Gradient clipping:** Model-dependent.
- **Mixed precision:** Not default.
- **Determinism:** `torch.use_deterministic_algorithms(True)`, cudnn.deterministic=True. [globals.py](globals.py) lines 42-47.
- **Seeds:** S=5 per topology (Plot 2); experiment_config seeds [100,200,300,400,500]. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) line 2392; [experiment_config.yaml](experiment_config.yaml).
- **Validation selection:** Best validation (load_best=True in EarlyStopping).
- **Final model:** Best validation epoch.

---

## 8. Capacity Accounting

- **Parameter count:** Masked weights exist as parameters but are zeroed in forward. [count_sparse_parameters.py](count_sparse_parameters.py): counts only non-zero mask entries for fair comparison.
- **Masking:** Reduces effective parameters (zero weights); full parameter count unchanged unless sparse implementation.
- **FLOPs:** Not explicitly estimated in codebase.
- **Runtime vs sparsity:** Not explicitly optimized; dense matmul with mask.
- **Matching:** Capacity controlled via fixed H and fixed total parameter structure; masking zeros weights but parameter tensors remain same size.

---

## 9. Compute Resources

- **GPU:** Not specified in code; PyTorch/CUDA if available.
- **CUDA/PyTorch:** torch==2.6.0 per [environment.yml](environment.yml).
- **Runtime:** Not systematically reported.
- **Storage:** Outputs in architecture_refinement/outputs/, evaluation/results/.
- **Parallel trials:** n_jobs=1 typical for TPE; Optuna supports parallel.

---

# C. Extended Results

---

## 10. Per-dataset / per-perturbation breakdowns

- Per-subject tables: Available in analysis outputs. [analyze_plot2_results.py](architecture_refinement/analyze_plot2_results.py).
- Separate by noise type: Yes (gaussian, dropout, eog, etc.).
- Statistical tests: Hierarchical bootstrap, 95% CI; paired comparisons (B−A, C−B). [analyze_plot2_results.py](architecture_refinement/analyze_plot2_results.py).

---

## 11. Metric–Robustness Analyses

- Correlations: Pearson/Spearman in analysis scripts.
- Per dataset or pooled: Configurable.
- Significance: Via bootstrap CI.
- Robustness: Not explicitly tested (e.g., removing extremes).

---

## 12. Full Search Traces

- Trials: M0 + batches up to M_max.
- Hypervolume: Computed in normalized (TE_res, sigma) or (TE, ORC) space; ref point (-0.05, -0.05). [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 623-656.
- Infeasible trials: Pruned; not in candidate pool.
- Best-so-far: Tracked via Pareto front and HV history.

---

# D. Ablations

---

## 13. Constraint Ablations

- Not systematically implemented. capacity_filter is optional (`--capacity_filter`).
- Spec mentions constraint removal for ablation; would require re-running with filter off.

---

## 14. Metric Set Ablations

- Not systematically implemented. TE_res and sigma are primary objectives; ablation would require code changes.

---

## 15. Generator Family Ablations

- modular_ws_flex vs plain_ws_flex available via `--generator_mode`. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 2495-2501.
- ER/random sparse: Not in main Plot 2 pipeline; paper3 has comparisons.

---

# E. Metrics and Normalization

---

## 16. TE_res normalization specifics

- **Reference set size:** M_ref (default 200). [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) line 2512.
- **Sampling:** Random feasible WS-Flex graphs, stratified by k (from degree_regimes). [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) `_compute_reference_bounds` lines 174-196.
- **μ_TE(k):** Mean TE per k from reference set; stored in mu_te_by_k.json. [run_plot2_proxy_viability.py](architecture_refinement/run_plot2_proxy_viability.py) lines 691-696.
- **TE_res:** TE - μ_TE(k). [metrics_te_orc.py](architecture_refinement/metrics_te_orc.py) lines 273-287.
- **Bounds:** te_res_lo, te_res_hi from q_lo=0.05, q_hi=0.95 quantiles. [run_plot2_proxy_viability.py](architecture_refinement/run_plot2_proxy_viability.py) lines 742-750.
- **Clipping:** Normalize to [0,1] with clip. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) `_normalize_fixed` lines 127-139.
- **Fixed across experiments:** Yes; frozen in proxy_viability output (frozen_bin_edges.json, mu_te_by_k.json).

---

# F. Feasibility Constraints and BO Integration

---

## 17. Infeasible trial handling

- **Pruned immediately:** Yes; `TrialPruned()` for disconnected/capacity violations. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 965, 973.
- **Logged:** Pruned trials in Optuna study; n_pruned in log.
- **Fed to surrogate:** No; pruned trials are not used for TPE model update.
- **Rejection rate:** Available from `rejection_diagnostics` in shared_random; TPE pruned count in log.

---

# G. Pareto Selection and Coverage

---

## 18. Pareto extraction

- **Pool:** Global pool of all candidates. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) `_pareto_membership_2d`.
- **Dominance:** Strict (no floating tolerance in standard Pareto).
- **Ties:** Handled by Pareto definition (non-dominated).

---

## 19. Coverage-aware selection

- **Regime quotas:** At least 2 per regime. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 1881-1892.
- **Bin quotas:** C/L tertile bins per regime; fixed from proxy viability. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) `_compute_cl_bins` lines 1187-1358.
- **Collapse cap:** max_per_cell = 50% of B. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) line 1595.
- **Bins:** Tertiles over (C, L) per regime; edges from frozen_bin_edges.json.
- **Deterministic ordering:** Yes; seed-controlled.

---

## 20. Edge Orientation Policy

- **Deterministic:** Yes.
- **Seeded:** wiring_seed = hash(graph_hash) mod 2^31-1. [run_plot2_topology_study.py](architecture_refinement/run_plot2_topology_study.py) lines 320-322.
- **Based on node index:** Indirectly; edge list order affects rng sequence.
- **Balanced in/out-degree:** No; random_oriented picks one direction per edge randomly.

---

## 21. CfC Masking Location

- **Mask multiplies W:** Yes. `F.linear(x, self.ff1.weight * self.sparsity_mask, ...)`. [ncps-master/ncps/torch/cfc_cell.py](ncps-master/ncps/torch/cfc_cell.py) lines 139-140, 153-154.
- **Before/after gating:** Before; applied to Linear outputs (ff1, ff2) before tanh/sigmoid.
- **Binary:** Yes (0/1).
- **Static:** Yes; requires_grad=False.

---

## 22. Readout

- **Final hidden state:** Typically last timestep or pooled.
- **Temporal mean pooling:** Model-dependent (e.g., HYDRA SSVEP head).
- **Logits per timestep:** Model-dependent.
- **Softmax:** At classification head.

---

# I. Capacity Matching Section

---

## 23. Parameter-count equations

- **CfC formula:** See [count_sparse_parameters.py](count_sparse_parameters.py). Input→hidden, hidden→hidden, hidden→output; mask applied to weight matrices.
- **Masked recurrent:** Count = nnz in wiring matrix for hidden block.
- **Input projection:** Dense (I×H) unless degree_proportional.
- **Output head:** Dense (H×O).
- **NCP partition:** NCP has different structure; see model definitions.

---

## 24. Capacity table

- Exact counts at H=32: Run `count_sparse_parameters.py` or equivalent for Dense CfC, Random Sparse, WS-Flex selected, NCP. Values depend on specific architectures; see [count_sparse_parameters.py](count_sparse_parameters.py) main().

---

## 25. Runtime

- **Wall-clock:** Not systematically reported.
- **Controlled:** Seeds control determinism; hardware varies.
- **Sparsity vs runtime:** Masked matmul same FLOPs as dense (zeros still computed); no sparse kernels.
- **Optimized:** Not explicitly.

---

# J. Perturbations and Evaluation

---

## 26. Evaluation protocol

- **Noise applied:** Test only (test_perturb mode); no noise during training for evaluation.
- **Calibration:** Per dataset/paradigm.
- **Clean retraining:** Model trained on clean data; evaluated on perturbed test.
- **Same model reused:** Yes; one trained model evaluated across intensity grid.

---

# K. Seeds and Statistics

---

## 27. Seeds

- **Count:** 5 per topology (Plot 2); 5 in experiment_config (100,200,300,400,500).
- **Fixed across models:** Per topology, seeds fixed for reproducibility.
- **Averaging:** Mean over seeds; hierarchical bootstrap for CIs.
- **Confidence intervals:** 95% CI via bootstrap.

---

# L. Non-EEG Validation

---

## 28. Non-EEG tasks

- **Status:** Not implemented. No MNIST, CIFAR, or other non-EEG datasets in codebase.
- **Generality claim:** Would require separate non-EEG experiments; currently EEG-only (MOABB).

---

# M. Global Consistency Checks

- **Duplicate sections:** Single instantiation path (arbitrary_wiring → WiredCfCCell).
- **Hyperparameters:** Documented in manifest, run_plot2_topology_study args, experiment_config.
- **Normalization equations:** TE_res = TE - μ_TE(k); sigma = (C/C_ER)/(L/L_ER); bounds from quantiles.
- **Feasibility:** Disconnected → pruned; capacity_filter optional; consistent in BO code.
- **Regime definitions:** degree_regimes in manifest; capacity_utils regime_bins; plot2_bounds_v2.yaml; consistent k-ranges.

---

# Final Note

Answers above cite specific files and line ranges. An external lab can reimplement by:

1. Using ws_flex_generator, metrics_te_orc, small_world_metrics, topology_analyzer.
2. Running run_plot2_topology_study with documented args.
3. Using arbitrary_wiring + ncps WiredCfCCell for graph→CfC.
4. Following augmentation/noise, utils, unified_experiment_runner for perturbations and evaluation.

All stochastic elements are seed-controlled (graph_seed, wiring_seed, training seeds, Optuna sampler seed).
