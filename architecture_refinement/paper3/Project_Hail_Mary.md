Below is a detailed experiment specification document for a **reframed Chapter 5 study** centered on whether recurrent topology induces measurable biases in **learnability, training stability, and perturbation sensitivity** under controlled conditions.

You can treat this as the working design document for the revised chapter.

---

# Experiment Specification Document

## Chapter 5 Reframing: Controlled Study of Topology-Induced Biases in Continuous-Time Recurrent EEG Decoders

## 1. Purpose of the Study

The original graph-metric NAS framing attempted to determine whether training-free graph descriptors could guide topology search toward more robust recurrent architectures after full training. The results did not support a strong or reliable claim of that type.

This revised study asks a narrower and more defensible question:

**Do recurrent topologies induce measurable biases in learnability, training stability, or perturbation sensitivity under controlled conditions?**

This reframing shifts the chapter away from a failed optimization narrative and toward a controlled mechanistic study of topology as an architectural factor.

The study is designed to answer three questions:

1. **Learnability:** Do some topologies make the model easier to optimize?
2. **Training stability:** Do some topologies produce more stable training dynamics?
3. **Sensitivity:** Do some topologies show different responses to controlled perturbations, even when final clean performance is similar?

---

# 2. Core Scientific Hypothesis

## Main hypothesis

Recurrent topology acts as an **inductive bias** that can influence the optimization trajectory and perturbation sensitivity of a continuous-time recurrent EEG model, even if simple graph proxies do not reliably predict final robustness after full training.

## Sub-hypotheses

### H1. Learnability

Different recurrent topologies will produce measurable differences in early-epoch validation performance, convergence speed, and seed variance under a fixed training recipe.

### H2. Training stability

Different recurrent topologies will produce measurable differences in training dynamics, such as gradient norm behavior, loss volatility, or hidden-state magnitude statistics.

### H3. Perturbation sensitivity

Different recurrent topologies will produce measurable differences in degradation under a controlled perturbation family, even when all other model components are held fixed.

---

# 3. Experimental Philosophy

This chapter should no longer be treated as a search problem. It should be treated as a **controlled ablation study**.

The design principle is:

**Freeze everything possible except the recurrent topology.**

That means:

* same dataset
* same protocol
* same feature extractor / model head
* same hidden dimension
* same optimizer and training hyperparameters
* same perturbation protocol
* same evaluation metrics
* same random seed handling framework

The topology is the experimental variable.

---

# 4. High-Level Study Design

## Independent variable

Recurrent hidden-to-hidden topology of the CfC/NCP-style recurrent block.

## Controlled variables

* Dataset
* Preprocessing pipeline
* Evaluation protocol
* Input window definition
* Feature extractor
* Hidden size
* Classifier head
* Training schedule
* Optimizer
* Learning rate
* Batch size
* Early stopping policy
* Perturbation intensity schedule
* Number of training epochs
* Random seed handling for data splits and perturbations

## Dependent variables

### Learnability outcomes

* Validation ROC-AUC at fixed early epochs
* Best validation ROC-AUC reached during training
* Epoch of best validation ROC-AUC
* Learning curve slope over early epochs
* Between-seed variance in early and final validation performance

### Stability outcomes

* Gradient norm statistics
* Batch-to-batch or epoch-to-epoch loss variance
* Hidden-state norm statistics
* Optional: logit variance under tiny input perturbation
* Optional: fraction of unstable or failed runs

### Sensitivity outcomes

* Clean test ROC-AUC
* Test ROC-AUC under low and moderate perturbation
* Relative degradation from clean to perturbed
* Optional: local degradation slope over 2 to 3 perturbation levels
* Optional: prediction consistency across repeated perturbation draws

---

# 5. Dataset and Evaluation Setting

## Recommended dataset

**BNCI2014_001**

This is the best default choice because:

* it is already integrated into your workflow
* it is familiar from prior experiments
* runtime characteristics are known
* subject/session organization is well understood
* it is a standard MI dataset with enough complexity to make the problem meaningful

## Recommended evaluation protocol

**Cross-session**

Reasoning:

* more realistic than within-session
* cheaper and more controlled than cross-subject
* already aligned with prior chapter work
* likely to preserve continuity with prior benchmark findings

## Recommended task formulation

Use the same task definition as in your prior Chapter 5 recurrent topology experiments. Do not introduce a new task or new preprocessing regime unless absolutely necessary.

---

# 6. Model Configuration

## Backbone

Use the same recurrent EEG decoder family as the NAS chapter, ideally the exact same model class if possible.

Most likely:

* CNN front-end or fixed feature extractor
* CfC/NCP-style recurrent block with masked hidden-to-hidden connectivity
* identical classifier head across all conditions

The point is to isolate the recurrent topology, not redesign the model.

## What must remain fixed

* hidden dimension
* recurrent cell implementation
* CfC parameterization
* residual/carry mechanism if one exists
* feature extraction path
* normalization layers
* classifier layers
* optimizer setup

## What changes across runs

Only:

* hidden-to-hidden adjacency mask / wiring topology
* training seed
* perturbation seed if repeated perturbation-draw sensitivity is included

---

# 7. Topology Panel Design

## Rationale

Do not sample hundreds or thousands of graphs. The goal is not to re-run NAS. The goal is to probe whether topology induces controlled behavioral differences.

You want a **small, intentionally selected topology panel** that spans meaningful structural regimes.

## Recommended topology family

Retain **WS-Flex** for continuity with the existing chapter, unless there is a compelling reason to replace it.

That allows you to say:

* the original search space is preserved
* the interpretation changes from optimization to controlled regime analysis

## Topology selection strategy

Select a small number of topologies spanning:

* sparse to dense
* regular to random
* low to high rewiring probability
* low to high local clustering / path-length variation
* low to high TE and/or |ORC| ranges if possible

## Suggested panel size

A practical target is:

* **8 to 12 topologies total**

A more conservative minimum:

* **6 topologies total**

## Example panel construction

Construct a structured regime grid over WS-Flex parameters such as:

* low (k), low (p)
* low (k), high (p)
* medium (k), low (p)
* medium (k), high (p)
* high (k), low (p)
* high (k), high (p)

Then, for each regime, instantiate either:

* one representative graph, or
* two graph seeds if budget allows

### Suggested concrete regime layout

For hidden size (H = 32), example degree levels might be:

* sparse: (k = 4)
* medium: (k = 10)
* dense: (k = 16)

Example rewiring levels:

* regular-ish: (p = 0.05)
* mixed: (p = 0.30)
* random-ish: (p = 0.80)

You do not need a full (3 \times 3) grid unless budget allows. Even a 6-point panel is sufficient if chosen carefully.

## Additional graph metadata to record

For each selected graph:

* (k)
* (p)
* graph seed(s)
* connectedness status
* number of edges
* degree distribution summary
* TE
* |ORC| average
* clustering coefficient
* average path length if available
* small-worldness if already implemented

These descriptors are now **descriptive covariates**, not optimization targets.

---

# 8. Experimental Conditions

There should be three main experiment blocks.

## Block A: Learnability study

Goal: determine whether topology affects optimization behavior and sample efficiency.

### Inputs

* Clean training data only
* No perturbation during training
* Standard validation split / subject-session split consistent with prior work

### Outputs

* validation ROC-AUC at selected epochs
* training loss curve
* validation loss curve
* best validation ROC-AUC
* epoch of best validation performance
* seed-wise variability

### Primary interpretation

Does topology affect how quickly and how reliably the model learns?

---

## Block B: Training stability study

Goal: determine whether topology affects internal training behavior.

### Inputs

Same training runs as Block A if possible. This should be instrumented, not a separate full experiment unless needed.

### Outputs

* gradient norm summaries
* epoch-level loss variance
* hidden-state norm summaries
* optional local sensitivity metric

### Primary interpretation

Does topology affect optimization smoothness or dynamical stability?

---

## Block C: Perturbation sensitivity study

Goal: determine whether topology affects response to controlled corruption.

### Inputs

Use the trained models from Block A, then evaluate under:

* clean
* low perturbation
* moderate perturbation

### Perturbation family

Use **one perturbation family only**.

#### Best default choice

Choose whichever of these best supports your broader dissertation logic:

### Option 1: EOG artifact

Use if you want to stay aligned with the idea that recurrence may be more meaningful under structured temporal perturbation.

### Option 2: Gaussian noise

Use if you want maximal implementation simplicity and fewer moving parts.

### Option 3: Channel dropout

Use only if your prior setup already supports it cleanly and you believe it is mechanistically informative for recurrent topology.

### Recommendation

For this chapter, **EOG or Gaussian** is preferable.

* EOG is more interesting mechanistically.
* Gaussian is cleaner operationally.

### Perturbation levels

Use only:

* clean
* low intensity
* moderate intensity

Do not run full perturbation curves unless runtime is trivial.

### Outputs

* test ROC-AUC under each level
* degradation from clean to low
* degradation from clean to moderate
* optional mini-AUPC across 3 points
* optional repeated-draw variance

### Primary interpretation

Does topology affect local perturbation sensitivity under controlled conditions?

---

# 9. Metrics and Operational Definitions

## 9.1 Learnability metrics

### Early validation performance

Validation ROC-AUC at fixed epochs, for example:

* epoch 5
* epoch 10
* epoch 20
* epoch 50 if training is long enough

This is one of the most important outcomes.

### Best validation performance

Maximum validation ROC-AUC observed during training.

### Convergence speed

Operationalize as one or more of:

* epoch of best validation ROC-AUC
* first epoch at which performance exceeds a threshold
* area under early learning curve

### Seed variance

Standard deviation across random seeds for:

* early validation ROC-AUC
* best validation ROC-AUC
* final test ROC-AUC

### Run failure / poor convergence rate

Count runs that:

* diverge
* plateau near chance
* fail to exceed minimal expected performance

This can be very informative if any topology regime is fragile.

---

## 9.2 Stability metrics

These should be cheap summaries, not an enormous new instrumentation project.

### Gradient norm statistics

Track per epoch:

* mean gradient norm
* max gradient norm
* variance of gradient norm
* optional separation by recurrent block vs whole model

If block-specific gradients are easy to isolate, recurrent-block gradients are especially valuable.

### Loss volatility

Track:

* variance of batch training loss within epoch
* variance of validation loss across epochs
* optional moving average smoothness metric

### Hidden-state norm statistics

For a fixed validation minibatch or subject:

* mean hidden-state norm
* max hidden-state norm
* variance across time steps or samples

This gives a coarse proxy for dynamical magnitude control.

### Optional: local input sensitivity

For a small input perturbation (\epsilon):

* compute average output/logit change
* or average prediction probability change

This is a cheap local sensitivity estimate if easy to implement.

---

## 9.3 Sensitivity metrics

### Clean ROC-AUC

Baseline test performance.

### Perturbed ROC-AUC

Performance at low and moderate perturbation intensity.

### Relative degradation

For performance (a_0) under clean input and (a_\delta) under perturbation:
[
RD_\delta = \frac{a_0 - a_\delta}{a_0}
]

### Local degradation slope

If using clean, low, moderate:
[
\text{slope}*{local} \approx \frac{a*{moderate} - a_{clean}}{\delta_{moderate} - 0}
]

### Mini-AUPC

If you want continuity with prior chapter metrics, compute a local trapezoidal area using the 3-point curve:

* clean
* low
* moderate

This should be explicitly framed as a reduced local robustness summary, not a full robustness benchmark.

---

# 10. Statistical Analysis Plan

The study should not be over-claimed. Use simple, transparent statistics.

## Primary level of analysis

Each topology is a condition. Each seed is a replicate. If subject-level results are available, subjects can serve as additional repeated measurements for some analyses.

## Recommended statistical summaries

For each metric report:

* mean
* standard deviation
* 95% confidence interval where practical

## Recommended inferential tests

### For comparing topology regimes

If sample count is small and assumptions are unclear:

* Kruskal-Wallis for omnibus regime comparison
* Dunn-style or pairwise rank tests if needed

If subject-level paired results are available and conditions are matched:

* repeated-measures tests can be used cautiously

### For correlation with descriptive graph metrics

Use:

* Spearman correlation
* optionally Pearson if clearly appropriate

But keep correlation claims modest.

### For mixed interpretation

If you have enough observations:

* simple linear regression or mixed effects is possible
* but do not make this a modeling chapter

The priority is clarity, not statistical sophistication.

---

# 11. Figures and Tables to Produce

## Essential figures

### Figure 1. Topology panel overview

A table or schematic showing selected topologies and their descriptors:

* topology ID
* (k)
* (p)
* edge count
* TE
* |ORC|
* clustering
* path length if available

This establishes the structural panel.

### Figure 2. Learning curves by topology

Plot validation ROC-AUC over epochs for each topology or topology regime.

If individual curves are too noisy:

* show mean across seeds with CI band
* or regime-aggregated curves

### Figure 3. Early learnability comparison

Bar plot or boxplot for epoch-10 / epoch-20 validation ROC-AUC by topology.

This may be the single most important plot in the chapter.

### Figure 4. Stability summary

Possible options:

* gradient norm summary by topology
* hidden-state norm summary by topology
* training loss volatility summary

Choose one or two, not all three if space is tight.

### Figure 5. Clean vs perturbed performance

For each topology:

* clean ROC-AUC
* low perturbation ROC-AUC
* moderate perturbation ROC-AUC

This can be shown as:

* paired line plot
* grouped bars
* degradation plot

### Figure 6. Descriptive proxy scatter

Scatter plots of:

* TE vs early learnability
* |ORC| vs degradation
* TE vs seed variance

These are now exploratory support figures, not the core of the chapter.

---

## Essential tables

### Table 1. Topology descriptors

One row per topology.

### Table 2. Learnability results

One row per topology, columns for:

* epoch-10 val AUC
* epoch-20 val AUC
* best val AUC
* epoch of best val
* seed std

### Table 3. Sensitivity results

One row per topology, columns for:

* clean test AUC
* low perturbation AUC
* moderate perturbation AUC
* RD(low)
* RD(moderate)

### Optional Table 4. Stability metrics

One row per topology, columns for:

* mean grad norm
* grad norm variance
* hidden-state norm mean
* loss volatility

---

# 12. What Can Likely Be Reused From the NAS Chapter

A large portion of the infrastructure should be reusable.

## Likely reusable components

### 12.1 Graph generation code

You likely already have:

* WS-Flex parameterized graph generator
* graph seed handling
* graph validity checks
* connectedness enforcement
* adjacency export

This should be reused directly.

### 12.2 Graph metric computation

You likely already have code for:

* TE
* ORC or average absolute ORC
* maybe clustering/path length/small-worldness

Reuse these as descriptive annotations.

### 12.3 Topology-to-mask conversion

You likely already have a function that:

* converts graph adjacency into recurrent mask matrices
* applies deterministic direction/orientation
* builds hidden-to-hidden wiring for the recurrent layer

This should be reused unchanged if possible.

### 12.4 Recurrent model implementation

Your masked CfC / NCP recurrent block should already exist.
Do not rewrite the core recurrent block unless necessary.

### 12.5 Training harness

You likely already have:

* experiment config system
* train/validation/test loop
* seed control
* logging
* checkpointing
* metric tracking

This is all reusable.

### 12.6 Benchmark evaluation utilities

Your robustness chapter likely already has:

* perturbation injection utilities
* evaluation loop for perturbed data
* AUC metric computation
* reproducible perturbation seeds

These should be repurposed for the reduced perturbation evaluation.

### 12.7 Subject/session splitting and preprocessing

All dataset loading and split logic should remain identical to prior runs.

---

# 13. What Needs to Be Implemented or Modified

This is the most important operational section.

## 13.1 New topology panel selection logic

Instead of search-driven topology proposal, you need a new function or config path that:

* takes a manually specified list of topology conditions
* generates one or more graphs for each condition
* assigns stable topology IDs
* stores descriptive metadata for each graph

### Required output

A structured manifest file, for example CSV or JSON, with fields like:

* topology_id
* hidden_size
* k
* p
* graph_seed
* valid_graph_flag
* num_edges
* TE
* ORC_abs_mean
* clustering
* path_length
* mask_file_path or serialized mask

This is essential.

---

## 13.2 Instrumentation for early-epoch metrics

Your current training code may already log validation metrics by epoch. If so, this is almost free.

You need to ensure you can reliably extract:

* val AUC at fixed epochs
* best val AUC
* epoch of best val
* train loss and val loss trajectories

### Implementation need

Possibly just:

* a postprocessing script to aggregate logs
* or a small callback that saves epoch-wise metrics in a structured file

---

## 13.3 Instrumentation for stability metrics

This is probably the main new implementation burden.

### Minimum viable instrumentation

Track at least one of:

* gradient norm per epoch
* loss variance per epoch

### Preferred implementation

At end of each training epoch, log:

* global gradient norm
* recurrent-block gradient norm
* mean batch loss
* variance of batch loss

### Optional hidden-state instrumentation

If your recurrent block can expose hidden states without major code surgery, add:

* mean hidden-state norm for one validation batch per epoch
* max hidden-state norm
* maybe variance across time

If this requires invasive model changes, it can be dropped.

---

## 13.4 Reduced perturbation evaluation mode

Your current robustness framework may expect full perturbation sweeps.
You need a lightweight evaluation mode that:

* loads trained checkpoint
* evaluates clean
* evaluates low perturbation
* evaluates moderate perturbation
* stores results in one tidy table

### Recommendation

Add a config flag for:

* `perturbation_levels = [0.0, 0.25, 0.50]` or similar
  rather than full dense curves

This should be straightforward if the perturbation pipeline is modular.

---

## 13.5 Result aggregation and analysis scripts

You need a clean aggregation pipeline that joins:

* topology descriptors
* training summaries
* stability summaries
* perturbation results

into one analysis-ready dataframe.

### Required outputs

At minimum:

* per-run table
* per-topology aggregated table
* plotting notebook or script

This is critical. The revised chapter depends on clean comparative analysis.

---

## 13.6 Run manifest and reproducibility bookkeeping

Because the whole point is controlled analysis, every run must be well indexed.

You need a run manifest that tracks:

* topology_id
* dataset
* protocol
* train seed
* graph seed
* perturbation seed(s)
* checkpoint path
* log path
* config hash or config path

If your old code already has this partially, extend it rather than starting over.

---

# 14. Suggested Directory / Artifact Structure

A clean file structure will help a lot.

Example:

* `topology_panel/`

  * `topology_manifest.csv`
  * `graph_<id>.pkl`
  * `mask_<id>.pt`

* `runs/`

  * `dataset_bnci2014_001_cross_session/`

    * `topo_<id>_seed_<s>/`

      * `config.yaml`
      * `train_log.csv`
      * `val_log.csv`
      * `checkpoint.pt`
      * `stability_metrics.csv`

* `evaluation/`

  * `clean_and_perturbed_results.csv`

* `analysis/`

  * `merged_run_table.csv`
  * `topology_summary_table.csv`
  * `figures/`

This will make the chapter easier to write and audit.

---

# 15. Recommended Minimal Experimental Budget

This is not a timeline, but it defines a realistic minimum scope.

## Minimum viable version

* 6 topologies
* 2 seeds each
* clean training only
* early-epoch metrics
* one perturbation family
* clean + 2 perturbation levels

This is the fallback version.

## Preferred version

* 8 to 12 topologies
* 3 seeds each
* early learnability metrics
* at least one stability metric
* one perturbation family
* clean + low + moderate evaluation

That would be enough for a credible chapter section.

---

# 16. Recommended Interpretation Framework

The wording of the chapter matters as much as the results.

## What the chapter should claim if results are modest

A safe and likely defensible claim is:

> Recurrent topology does not yield a simple training-free proxy objective for final robustness, but it does induce measurable differences in learnability, optimization behavior, and local perturbation sensitivity under controlled conditions.

## What the chapter should not claim

Do not claim:

* topology metrics can optimize robustness
* TE or ORC are strong predictors of post-training performance
* a global best topology family was found
* graph search solved robustness

## If effects are weak

That is still useful. Then the chapter can say:

* topology effects are detectable but modest
* topology alone is insufficient to determine post-training robustness
* robustness emerges from topology interacting with learned dynamics and perturbation structure

That directly reinforces the broader dissertation thesis.

---

# 17. Example Experimental Workflow

## Phase 1. Build topology panel

1. Choose fixed hidden size.
2. Define structured WS-Flex regime grid.
3. Generate graph instances.
4. compute graph descriptors.
5. serialize masks and manifest.

## Phase 2. Train controlled runs

For each topology and seed:

1. initialize same model class
2. load same dataset/split protocol
3. apply topology-specific recurrent mask
4. train with same hyperparameters
5. log epoch metrics
6. log stability summaries
7. save checkpoint

## Phase 3. Evaluate sensitivity

For each trained checkpoint:

1. evaluate on clean test data
2. evaluate on low perturbation
3. evaluate on moderate perturbation
4. compute RD and optional local AUPC
5. store results

## Phase 4. Aggregate and analyze

1. merge topology metadata with training and evaluation results
2. compute per-topology means and CIs
3. generate figures and tables
4. test simple correlations and regime comparisons
5. write chapter around controlled effects, not optimization

---

# 18. Concrete Deliverables

At the end of this effort, the revised chapter should be supported by the following concrete artifacts:

## Data artifacts

* topology manifest
* per-run training logs
* per-run stability logs
* clean and perturbed evaluation table
* merged analysis table

## Figures

* topology panel figure
* learning curve figure
* early-epoch performance comparison
* stability summary figure
* clean vs perturbed comparison
* optional TE/ORC exploratory scatter

## Tables

* topology descriptor table
* learnability results table
* perturbation sensitivity table
* optional stability table

## Textual outcomes

* revised chapter motivation
* methodology subsection for controlled topology study
* results subsection for learnability
* results subsection for sensitivity/stability
* discussion subsection on why proxy optimization failed but topology still matters

---

# 19. Final Recommendation

The strongest version of this experiment set is the one that stays disciplined.

Keep the scope narrow. Reuse the existing NAS infrastructure wherever possible. Implement only the instrumentation needed to expose **proximal effects** of topology. Treat graph metrics as descriptive variables, not optimization objectives. Build the chapter around the conclusion that topology may matter through **local and mechanistic biases**, not through a simple training-free path to robust architecture search.

That gives the chapter a tighter and more credible role in the dissertation:

* Chapter 3 shows robustness is nuanced and depends on architecture, perturbation, and signal structure.
* Chapter 4 shows that careful recurrent architectural design can improve robustness.
* Chapter 5 shows that topology alone does not provide a simple optimization handle, but it does leave measurable signatures on training dynamics and perturbation response.

If you want, I can turn this into a more formal lab-style protocol with exact config fields, pseudocode for the run loop, and a checklist of implementation tasks grouped by code module.
