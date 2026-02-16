====================================================================
Plot 2 Diagnostic Mini Run Specification — Coverage & Saturation Check
====================================================================

Purpose
-------
This diagnostic mini run is a **training-free, graph-only experiment**
designed to answer a single critical question before any further model
training:

    “Does the current WS-Flex (k, p, connectivity-only) generator
     actually produce sufficient diversity in graph-measure space
     (clustering C, path length L) to justify proxy-guided search?”

This mini run exists to **separate algorithmic limitations from
search-space limitations**.

It is explicitly motivated by:
- The Plot 2 mini-scale training results, where Random WS-Flex ≈ TPE WS-Flex.
- The observed collapse of ORC and TE onto degree/density.
- The methodological gap relative to Waqas et al. and You et al., who
  treat (C, L) coverage as a primary experimental control.

This run requires **no model training** and should complete in minutes.

====================================================================
Key Questions This Mini Run Must Answer
====================================================================

Q1) Given current bounds and pruning (k, p, connectivity-only), how much
    diversity in (C, L) space does WS-Flex actually generate?

Q2) Does TPE meaningfully bias sampling toward different (C, L) regions
    than random sampling, *before any selection is applied*?

Q3) Is the observed saturation in Plot 2 caused by:
    (a) insufficient generator diversity, or
    (b) selection collapse only?

The outcome determines whether coverage-aware selection is sufficient,
or whether the generator/search space itself must be expanded.

====================================================================
Scope and Constraints
====================================================================

- NO model training
- NO robustness evaluation
- NO perturbations
- Graph generation + topology analysis only

This run is **purely diagnostic** and does not alter any Plot 2 results.

====================================================================
Experimental Design
====================================================================

1) Graph Generator (unchanged)
------------------------------
- Generator: WS-Flex
- Node count: H = 32 (locked)
- Degree k: sampled from degree regimes
    super_sparse: [2, 6]
    sparse:       [7, 12]
    moderate:     [13, 18]
    near_dense:   [19, 26]
- k must be even and 2 ≤ k ≤ H−2
- Rewiring probability p ~ Uniform(0, 1)
- Connectivity constraint: nx.is_connected(G)
- No constraints on clustering or path length

2) Sampling Budgets
-------------------
- Random WS-Flex pool:
    N_random = 2000 graphs
- TPE WS-Flex pool:
    N_tpe = 2000 trials

Budgets may be reduced to 1000 if compute is constrained, but random and
TPE must match exactly.

3) TPE Configuration
--------------------
- Objectives:
    maximize TE
    maximize ORC
- Sampler: Optuna TPE
- No early stopping
- No selection / Pareto truncation
- All evaluated trials retained for analysis

====================================================================
Computed Metrics (Per Graph)
====================================================================

For every generated graph, compute and store:

- Degree k
- Edge density
- Clustering coefficient C
- Average path length L
- Topological entropy (TE)
- Average Ollivier–Ricci curvature (ORC)

Store results in a single table per pool:
- random_ws_flex_metrics.csv
- tpe_ws_flex_metrics.csv

====================================================================
Analysis Procedures
====================================================================

A) Distribution Analysis
------------------------
For random and TPE pools separately:
- Plot or compute distributions of:
    - C
    - L
    - (C, L) joint scatter
- Stratify plots by degree regime

B) Coverage Quantification
--------------------------
Within each degree regime:

1) Define bins:
   - C_bins: tertiles (low / mid / high)
   - L_bins: tertiles (low / mid / high)

2) Compute:
   - total possible bins = 9
   - occupied bins count
   - coverage_score = occupied_bins / 9

3) Compute occupancy histogram:
   - number of graphs per (C_bin, L_bin)

C) TPE vs Random Comparison
---------------------------
For each regime:
- Compare coverage_score_random vs coverage_score_tpe
- Compare marginal distributions of C and L (e.g., KS statistic)
- Compare correlation matrices:
    corr(k, C), corr(k, L), corr(ORC, C), corr(ORC, L)

====================================================================
Success / Failure Criteria
====================================================================

This mini run is considered successful if it clearly resolves one of the
following outcomes:

Outcome 1 — Generator is sufficient:
- coverage_score ≥ 0.6 in ≥2 regimes for random pool
- TPE shows biased occupancy toward distinct (C, L) regions
→ Coverage-aware selection alone is likely sufficient.

Outcome 2 — Generator is insufficient:
- coverage_score ≤ 0.4 in most regimes for both pools
- TPE and random distributions in (C, L) are nearly identical
→ Generator/search space must be expanded (additional knobs required).

Outcome 3 — Mixed:
- Some regimes show good coverage, others collapse
→ Hybrid solution: coverage-aware selection + regime-specific generator
  tweaks.

Ambiguous outcomes (e.g., marginal differences everywhere) should be
treated as Outcome 2.

====================================================================
Required Outputs
====================================================================

- random_ws_flex_metrics.csv
- tpe_ws_flex_metrics.csv
- diagnostic_summary.json containing:
    - coverage_score per regime and pool
    - occupied bin counts
    - key correlations
    - KS statistics (optional but recommended)

- One summary table (human-readable):
    Regime | Pool | Coverage | Dominant bins | Notes

====================================================================
How This Informs Plot 2
====================================================================

- If Outcome 1:
    Proceed with the coverage-aware selection patch (v2) and re-run
    Plot 2 mini-scale training.

- If Outcome 2:
    Pause Plot 2 and redesign the search space:
    e.g., introduce soft (C, L) objectives, directed edges, or additional
    WS-Flex control parameters.

- If Outcome 3:
    Apply coverage-aware selection selectively and document limitations
    explicitly in the paper.

====================================================================
Rationale
====================================================================

This diagnostic mirrors the philosophy of Waqas et al. and You et al.:
**search space characterization precedes model evaluation**.

Running this experiment avoids expensive training cycles and ensures
that subsequent Plot 2 results reflect algorithmic merit rather than
latent generator collapse.

====================================================================
End of Diagnostic Mini Run Specification
====================================================================
