# Architecture Refinement for Robustness-Aware CfC Networks — Detailed Summary

## Purpose and Novelty
- **Goal**: Select and wire recurrent units for CfC-based EEG models using a training-free search guided by graph-theoretic properties, then deploy the best wiring(s) directly into trainable models.
- **Why it works for our use-case**: EEG tasks benefit from architectures with balanced local clustering, long-range integration, redundancy, and robustness to noisy channels. These are precisely the structural traits quantifiable by graph metrics (e.g., clustering, path lengths, curvature, spectral connectivity). By optimizing those metrics up front, we bias the search toward biologically plausible, noise-tolerant recurrent fabrics before any gradient-based training.
- **Novelty in our setting**: Prior training-free NAS via graph metrics exists, but our contribution is applying it to CfC/NCP-style recurrent backbones and integrating the resulting WS-flex graph fabrics into EEG pipelines, including direct conversion to `WiredCfC`-compatible models and evaluation on MOABB paradigms.

## System Overview
The pipeline consists of four major stages that operate before and after learning:
1) Graph generation (candidate recurrent fabrics)
2) Pre-training topological analysis (compute metrics)
3) Multi-objective selection (Optuna over metric space)
4) Conversion to trainable models (Wired CfC / ArbitraryWiring)

Key modules and where they live:
- `architecture_refinement/graph_generator.py`: Modular small-world candidate generation (WS-flex variants, tunable degree, rewiring, clustering/path targets).
- `architecture_refinement/topology_analyzer.py`: Batch computation of metrics (entropy, Ricci curvature, algebraic connectivity, efficiency, clustering, paths, spectral measures) and composite robustness scoring.
- `architecture_refinement/optimizer.py`: Multi-objective search using Optuna over generator hyperparameters; records metrics, graphs, and selected solutions.
- `architecture_refinement/architecture_converter.py`: Utilities for transforming selected graphs to model-ready wiring specs.
- `architecture_refinement/arbitrary_wiring.py`: `WsFlexHiddenWiring` and `ArbitraryWiring` to ingest hidden-only WS-flex graphs and produce full NCP-compatible matrices (Input–Hidden–Output) under controllable policies.
- `models/cnnncp.py`: `CNNWiredCfC` model supporting arbitrary wiring; `CNNSmallWorld` for baseline small-world.
- `models/branched_wiredcfc.py`: Branched variant combining DIVA-style front end with parallel CfC branches driven by `ArbitraryWiring`.
- `architecture_refinement/integrate_with_evaluation.py`: Glue for instantiating `CNNWiredCfC` from saved architectures and running inside the unified evaluation runner.

## Stage 1: Candidate Graph Generation
- Family: Watts–Strogatz flexible (WS-flex) and modular small-world variants.
- Controls: number of units, mean degree k, rewiring probability p, optional targets for clustering and path length; module community structure in modular variants.
- Rationale for EEG: Small-world graphs balance short paths (global integration) with clustering (local specialization), a structure frequently linked to brain networks.

## Stage 2: Pre-Training Topological Analysis
Implemented in `TopologyAnalyzer`:
- **Basic**: node/edge counts, density, degree statistics.
- **Entropy**: degree entropy; weight/path length distributions where available.
- **Curvature**: Ollivier–Ricci (and options for Forman), proxies for flow robustness and bottleneck fragility.
- **Connectivity**: algebraic connectivity (Fiedler value), edge/node connectivity, expansion.
- **Efficiency**: global/local efficiency, cost efficiency.
- **Clustering/Paths**: average clustering, characteristic path length.
- **Spectral**: Laplacian spectrum-derived measures.
- Output can be combined into a **robustness score** capturing resilience, redundancy, and efficient communication.

Why these metrics help before training:
- They capture inductive biases that gradient descent cannot easily add later (e.g., multi-path redundancy or expansion properties) while remaining task-agnostic, fast, and reproducible.

## Stage 3: Multi-Objective Optimization
Implemented in `MultiObjectiveOptimizer`:
- Uses Optuna to sweep generator parameters and evaluate candidate graphs by metric objectives (e.g., maximize entropy proxy, curvature magnitude, algebraic connectivity, efficiency; optionally include modularity and redundancy).
- Returns a set of Pareto-favorable solutions with their graphs and metric tables.
- Reproducibility: controlled seeds at generator and study levels. In the broader experiments, a single global seed is passed by the harness [[memory:6545297]].

## Stage 4: Conversion to Trainable Models
Two complementary paths are supported:
1) `WsFlexHiddenWiring` builds a full Input–Hidden–Output wiring from a hidden-only WS-flex adjacency using explicit policies:
   - Inputs→Hidden: `dense` or `degree_proportional` fan-in.
   - Hidden↔Hidden: the oriented WS-flex fabric, with options: `symmetric`, `random_oriented`, or `as_is`; optional signed edges (inhibitory ratio) and self-loops.
   - Hidden→Outputs: `dense` or `uniform` fan-in.
   - Produces an `ArbitraryWiring` instance and complete wiring matrix consistent with NCP/CfC expectations.
2) Direct `ArbitraryWiring`: accepts a full matrix (I+H+O)×(I+H+O) when already constructed upstream, validates it, and exposes layer-wise neuron groups and synapses.

Resulting wiring integrates with:
- `CNNWiredCfC` in `models/cnnncp.py` for standard pipelines.
- `BranchedWiredCfC` in `models/branched_wiredcfc.py` for multi-branch temporal processing with attention fusion.

## Why CfC Backbones
- CfC provides continuous-time dynamics with strong sequence modeling under efficient parameterization, which pairs well with sparse or structured connectivity.
- The graph-derived fabric imposes pathway structure (e.g., community clusters with long-range hubs), complementing CfC’s dynamics and improving robustness to channel corruption and nonstationarities typical in EEG.

## Integration with EEG/MOABB
- Architectures from the search are serialized as JSON (hidden adjacency or full wiring specs) in `outputs/architectures/`.
- `architecture_refinement/integrate_with_evaluation.py` loads an architecture and instantiates `CNNWiredCfC` with dataset-specific shapes (channels, time points, class count), ready for the unified experiment runner.
- The global experiment system includes baselines (EEGNet, REEGNet, CNNSmallWorld, CfC/NCP variants) to compare against architecture-refined CfC models under clean and corrupted conditions.

## Reproducibility and Determinism
- Seeds are passed from the experiment CLI; graph generation, orientation, and randomized policies respect the provided seed. Use the single harness seed throughout [[memory:6545297]].
- Cached outputs and logging include metric tables and chosen hyperparameters to enable exact regeneration.

## Practical Usage Patterns
- Quick demo: see `architecture_refinement/README.md` for a short run via `demo.py` (reduced trials).
- Full runs: increase trials/timeouts in `MultiObjectiveOptimizer`, save top-k graphs, and convert to `CNNWiredCfC` for evaluation.
- Wiring policies: start with `input_strategy="degree_proportional"` and `output_strategy="uniform"` to emphasize selective fan-in and distributed readout; optionally enable signed hidden edges for inhibition.

## Empirical Expectations for EEG
- Improved robustness to channel dropouts and artifact noise due to redundant, diverse paths and higher algebraic connectivity.
- Better generalization across sessions/subjects where nonstationarity is present, thanks to balanced integration and modular clustering.
- Interpretability gains via community-aligned substructures and explicit edge policies.

## Limitations and Considerations
- Metric proxies are task-agnostic; Pareto-optimal graphs by metrics may still underperform for niche paradigms—retain downstream validation.
- Some metrics (e.g., Ricci curvature) are computationally heavy for very large H; consider batched or approximate computations.
- Wiring policies can influence results as much as the hidden fabric—document chosen policies and seeds in reports.

## For the Presentation and Paper
- Positioning: training-free NAS via graph metrics as a structural prior for CfC EEG models—fast, interpretable, and robust.
- Methods figure: pipeline boxes for generation → metrics → multi-objective selection → wiring conversion → model.
- Table: list of metrics and intuitions (robustness, redundancy, integration, efficiency).
- Ablations: compare wiring policies, signed vs. unsigned edges, symmetric vs. oriented hidden blocks, branched vs. single-stream models.
- Reproducibility: single-seed execution, saved JSON architectures, and integration hooks enabling exact reruns.

## Pointers to Code
- Topology metrics: `architecture_refinement/topology_analyzer.py`
- Optimization loop: `architecture_refinement/optimizer.py`
- Wiring builders: `architecture_refinement/arbitrary_wiring.py` (`WsFlexHiddenWiring`, `ArbitraryWiring`)
- Model classes: `models/cnnncp.py` (`CNNWiredCfC`), `models/branched_wiredcfc.py`
- Integration script: `architecture_refinement/integrate_with_evaluation.py`

---
This document goes beyond the existing README to explain why the training-free, graph-metric-guided search is well-suited for CfC-based EEG models, how the system is organized, and how to apply it in practice for experiments and publication.
