# Plot 2 decisive experiment specification and pivot plan

## Executive summary

Your current mini-scale Plot 2 configuration is structurally set up to *reduce separability* between TPE-selected and random-selected WS-Flex graphs, even if a real advantage exists. The key reason is the **coverage-aware selection policy** (`selection_coverage_level=regime_cl_bins_fixed`), which intentionally spreads selections across regimes and (C,L) bins, making different selectors converge toward similar portfolios. The small number of training seeds (**S=1**) further increases variance and makes meaningful deltas hard to detect.

A decisive answer requires two things:

First, a **clean ablation of the selection constraint**: compare TPE vs random under (a) **regime-stratified** coverage only (controls degree regime; still allows optimization) and (b) **fully unconstrained** selection (tests raw “best-of-basin” capability).  

Second, a **proxy validity gate** before heavy training: if no reasonable proxy (or composite proxy) predicts robustness better than chance on a small labeled set, then TPE cannot be expected to beat random at scale, and the right move is the pivot.

The final experiment spec below is modular, includes precise graph-metric equations (including TE variants, ORC, spectral and Laplacian measures, effective resistance, motifs, centralities, and robustness-normalized targets), prescribes lightweight proxy-only sub-experiments to quantify predictive power, and defines heavy training runs with exact go/no-go thresholds and an explicit pivot plan.

## Why the current mini-scale Plot 2 struggles to show separation

The WS-Flex design-space approach used in “graph-structured neural networks” work commonly emphasizes **coverage of clustering coefficient and average path length** (C and L) because performance often varies smoothly over that space; a “sweet spot” region can exist. This concept is well-documented in relational-graph studies of neural networks, where performance is approximately a smooth function of clustering and path length and top-performing graphs cluster in a region rather than being isolated points. [R2]

However, that same coverage principle becomes a problem when the goal is to show that one selector (TPE) reliably outperforms another (random). If the protocol forces each method to pick roughly the same number of graphs per regime and per (C,L)-bin, then even a strong optimizer cannot allocate most of its budget to the best region. The design becomes a **landscape survey** rather than a **selector head-to-head**.

Separately, many graph-theoretic “robustness proxies” are known to correlate with robustness only after controlling for confounders like density/degree and task scale. For example, the architectural robustness literature finds that certain structural measures (including curvature and entropy) can correlate with robustness, with stronger correlation on larger models and harder tasks. [R1] In your mini-scale setting (3 subjects and S=1), those correlations can be noisy, and a subtle selection advantage may be indistinguishable from training variance.

These two effects (protocol-constrained selection + underpowered training replication) are sufficient to explain “everything looks the same” outcomes even if the underlying basin contains gradients that TPE could exploit.

## Graph metrics and proxy suite with precise equations and normalization

This section defines a **canonical feature set** for WS-Flex hidden graphs and oriented hidden graphs. The intent is not to use all metrics in the final selection objective; it is to (a) test predictive power, then (b) shrink to a minimal, defensible proxy set.

### Graph objects and notation

Let \(G=(V,E)\) be the hidden-node graph with \(n=|V|=H\). Define:

- Undirected adjacency \(A \in \{0,1\}^{n\times n}\), symmetric, with \(A_{ii}=0\).
- Directed/oriented adjacency \(\widetilde{A} \in \{0,1\}^{n\times n}\) (orientation rule fixed by a deterministic seed per graph; see spec for reproducibility).
- Degree \(d_i = \sum_j A_{ij}\); directed out-/in-degrees \(d^{out}_i=\sum_j \widetilde{A}_{ij}\), \(d^{in}_i=\sum_j \widetilde{A}_{ji}\).
- Degree matrix \(D=\mathrm{diag}(d)\), Laplacian \(L = D - A\).
- Normalized Laplacian \(L_{norm} = I - D^{-1/2} A D^{-1/2}\) for graphs without isolated nodes.

Where a metric is undefined for disconnected graphs, the spec enforces connectivity (or uses the giant component).

### Structural metrics known to be relevant in architecture studies

These are widely used to span the WS(-flex) design space and relate structure to network behavior. [R2]

**Clustering coefficient (local and mean)**  
For node \(i\), let \(t_i\) be the number of triangles touching \(i\).  
\[
C_i = \frac{2 t_i}{d_i(d_i-1)}\quad \text{(for } d_i\ge 2\text{)}, \qquad C=\frac{1}{n}\sum_i C_i.
\]
(Equivalent forms appear in network-science robustness/architecture papers.) [R1]

**Average shortest path length**  
Let \(\mathrm{dist}(i,j)\) be the shortest-path distance in \(G\).  
\[
L = \frac{1}{n(n-1)}\sum_{i\ne j}\mathrm{dist}(i,j).
\]

**Small-worldness \(\sigma\)**  
A standard sigma definition used in small-world analysis is:  
\[
\sigma = \frac{C/C_{\mathrm{rand}}}{L/L_{\mathrm{rand}}}
\]
where \(C_{\mathrm{rand}}\) and \(L_{\mathrm{rand}}\) come from an appropriate random-graph baseline matched on \(n\) and edge density (or degree sequence). This definition is standard in small-worldness benchmarking. [R3]

### Entropy proxies (two variants) and residualization

“Topological entropy” is not uniquely defined across literatures. Two defensible variants:

**Spectral topological entropy (as used in robustness-architecture work)**  
One robustness-architecture study defines graph entropy via the spectral radius of adjacency:  
\[
H_{\mathrm{spec}} = \log(\rho(A)),\qquad \rho(A)=\max_k |\lambda_k(A)|.
\]
This is explicitly defined as “log of the spectral radius of the adjacency matrix” in that work. [R1]

**Degree-distribution entropy (common, normalized Shannon entropy)**  
Define a probability mass on nodes proportional to degree: \(p_i = d_i/\sum_j d_j = d_i/(2|E|)\).  
\[
H_{\mathrm{deg}} = -\sum_{i=1}^n p_i \log p_i,
\qquad 
\mathrm{TE} = \frac{H_{\mathrm{deg}}}{\log n} \in [0,1].
\]
This version is often used when *degree heterogeneity* is the meaningful property (and it is cheap/stable at small \(n\)).

**Residualization to form TE\_res**  
If selection varies \(k\) (degree regime), entropy and many metrics become trivially \(k\)-dependent. Define a residualized proxy:
\[
\mathrm{TE\_res}(G) = \mathrm{TE}(G) - \mu_{\mathrm{TE}}(k),
\]
where \(\mu_{\mathrm{TE}}(k)\) is the mean \(\mathrm{TE}\) over a reference sample of graphs with that \(k\). A stronger version uses z-scores:
\[
\mathrm{TE\_z}(G) = \frac{\mathrm{TE}(G) - \mu_{\mathrm{TE}}(k)}{\sigma_{\mathrm{TE}}(k)+\epsilon}.
\]
**Normalization choice:** use TE\_z for modeling and TE\_res for plotting; clip TE\_z to \([-5,5]\) to limit outliers.

### Curvature proxies

**Ollivier–Ricci curvature (ORC) on graphs**  
For adjacent nodes \(x,y\) with distance \(d(x,y)=1\), define probability measures \(p_x, p_y\) over nodes, typically:
- \(p_x(x)=\alpha\) (idleness), and for neighbors \(z\in N(x)\), \(p_x(z)=(1-\alpha)/d_x\).

Then ORC along \((x,y)\) is:
\[
\kappa(x,y)= 1 - \frac{W_1(p_x,p_y)}{d(x,y)} = 1 - W_1(p_x,p_y),
\]
where \(W_1\) is Wasserstein-1 (earth mover’s distance) with ground distance given by graph shortest-path distance. This is the standard definition used in network robustness contexts and stated explicitly in robustness-architecture work. [R1]

Aggregate edge curvature as:
\[
\mathrm{ORC\_mean}(G) = \frac{1}{|E|}\sum_{(x,y)\in E}\kappa(x,y),
\quad 
\mathrm{ORC\_min}(G)=\min_{(x,y)\in E}\kappa(x,y).
\]

**Residualization to form ORC\_res**  
\[
\mathrm{ORC\_res}(G) = \mathrm{ORC\_mean}(G) - \mu_{\mathrm{ORC}}(k),
\]
or analogous z-scoring within \(k\).

**Why include curvature at all:** curvature has been used as a *pre-training* structural correlate of robustness in neural architecture settings, and it is one of the few graph measures explicitly claimed as pre-training robustness-relevant in that line of work. [R1]

### Spectral and Laplacian metrics

These capture global connectivity, expansion, and dynamical mixing properties.

**Spectral radius (undirected and directed)**  
\[
\rho(A)=\max_k|\lambda_k(A)|,\qquad \rho(\widetilde{A})=\max_k |\lambda_k(\widetilde{A})|.
\]
Use \(\rho(\widetilde{A})\) when the model’s effective dynamics depend on directed propagation; use \(\rho(A)\) when topology is treated as undirected.

**Adjacency spectral gap**  
Let \(\lambda_1\ge \lambda_2\ge \dots\) be eigenvalues of \(A\).  
\[
\mathrm{gap}_A = \lambda_1 - \lambda_2.
\]
Normalization: \(\mathrm{gap}_A/\lambda_1\) to remove scale with density.

**Algebraic connectivity (Fiedler value)**  
Let \(0=\lambda_1(L)\le \lambda_2(L)\le \dots\le \lambda_n(L)\).  
\[
a(G)=\lambda_2(L).
\]
For comparability across densities:
\[
a_{norm}(G)=\frac{\lambda_2(L)}{\lambda_n(L)+\epsilon}.
\]

**Normalized Laplacian spectral gap**  
Let \(0=\lambda_1(L_{norm})\le \dots\le \lambda_n(L_{norm})\le 2\).  
\[
\mathrm{gap}_{L_{norm}} = \lambda_2(L_{norm}).
\]

### Effective resistance and Kirchhoff index

Effective resistance is a classic robustness-related measure for connectivity and redundancy of paths.

Compute the Moore–Penrose pseudoinverse \(L^+\) of the Laplacian \(L\). Then effective resistance between nodes \(i,j\) is:
\[
R_{ij} = L^+_{ii} + L^+_{jj} - 2L^+_{ij}.
\]
Global summaries:
\[
K_f(G)=\sum_{i<j} R_{ij}= n\cdot \mathrm{tr}(L^+),
\qquad 
\overline{R}(G)=\frac{2K_f(G)}{n(n-1)}.
\]
Normalization: report \(\overline{R}\) and \(K_f/n^2\).

### Centrality distributions and summaries

Centralities often correlate with bottlenecks and vulnerability to targeted disruptions.

**Betweenness centrality**  
Let \(\sigma_{st}\) be the number of shortest paths between \(s\) and \(t\), and \(\sigma_{st}(v)\) the number passing through \(v\).  
\[
BC(v)=\sum_{s\ne v\ne t}\frac{\sigma_{st}(v)}{\sigma_{st}}.
\]

**Closeness centrality**  
\[
CC(v)=\frac{n-1}{\sum_{u\ne v}\mathrm{dist}(u,v)}.
\]

**Eigenvector centrality**  
\(c\) satisfies \(c=\frac{1}{\lambda}Ac\), usually taking the principal eigenvector.

**Summaries to use as graph features**  
For a node metric \(m(v)\), define:
- \(\mathrm{mean}(m)\), \(\mathrm{std}(m)\), \(\max(m)\)
- inequality via Gini:
\[
\mathrm{Gini}(m)=\frac{\sum_{i}\sum_j |m_i-m_j|}{2n\sum_i m_i+\epsilon}.
\]

Residualize these summaries by \(k\) where necessary.

### Assortativity and path redundancy

**Degree assortativity (Newman coefficient)**  
Let each edge connect endpoints with degrees \(j\) and \(k\) (here letters are degrees, not the WS parameter). Then assortativity is the Pearson correlation of degrees over edge endpoints:
\[
r = \frac{\sum_e (j_e-\mu)(k_e-\mu)}{\sum_e (j_e-\mu)^2+\epsilon},
\]
with \(\mu\) appropriately defined over stubs/endpoints.

**Edge/vertex connectivity (optional)**  
- Vertex connectivity \(\kappa_v(G)\): min nodes removed to disconnect.
- Edge connectivity \(\kappa_e(G)\): min edges removed to disconnect.  
At \(n=32\), these are feasible but can be slower; use approximations if needed.

### Motif counts

Motifs can capture local feedback/triadic structure beyond clustering.

**Triangle count (undirected)**  
\[
T=\frac{\mathrm{tr}(A^3)}{6}.
\]
Use density-normalized: \(T / \binom{n}{3}\).

**4-cycle count (undirected, optional)**  
\[
C_4=\frac{\mathrm{tr}(A^4)-2\sum_i d_i(d_i-1)-2|E|}{8}
\]
(then normalize by \(\binom{n}{4}\)). This formula assumes simple undirected graphs.

**Directed motifs (if \(\widetilde{A}\) is used)**  
If orientation matters, count canonical 3-node directed motifs (including feed-forward loops) using standard motif enumeration. Because \(n\) is small, brute enumeration is feasible.

### Candidate composite proxies

Define a standardized feature vector \(z(G)\) from the subset of residualized metrics (e.g., TE\_z, ORC\_z, \(\log \rho(\widetilde{A})\), \(\sigma_z\), \(a_{norm}\), \(\overline{R}_z\), etc.). Then:

**Linear combination (hand-designed or fitted)**  
\[
P_{\mathrm{lin}}(G)=w^\top z(G).
\]
Weights:
- start with equal weights,
- then fit \(w\) using ridge regression to predict robustness labels in the pilot module.

**PCA proxy**  
Compute PCA on \(z(G)\) over the unlabeled pool. Let \(u_1\) be the first component. Use:
\[
P_{\mathrm{pca}}(G)=u_1^\top z(G),
\]
or pick the component with maximum absolute correlation to robustness labels on the pilot set.

**Learned surrogate**  
Train a regression model \(f_\theta(z)\approx y(G)\), where \(y(G)\) is a robustness label (defined below). Candidates:
- ridge / elastic net (interpretable),
- gradient-boosted trees (strong for tabular),
- random forest (robust baseline).

Model selection: nested cross-validation on the pilot labeled set; report \(R^2\), Spearman correlation, and calibration error.

### Table of metrics, compute routes, and confounds

| Metric family | Concrete metric | Compute from | Typical cost at H=32 | Major confounds | Normalization / residualization |
|---|---|---:|---:|---|---|
| Small-world | \(C, L, \sigma\) | \(A\) | low–medium | degree/density, disconnectedness | within-\(k\) z-score; enforce connectivity |
| Entropy | \(\mathrm{TE}\) (degree entropy) | \(A\) | low | degree regime | TE\_res or TE\_z within \(k\) |
| Spectral entropy | \(H_{spec}=\log\rho(A)\) | \(A\) | low | density, \(k\) | use \(\log\rho\) then residualize within \(k\) |
| Curvature | ORC mean/min | \(A\) + shortest paths | medium | degree, computation choices (α, measure) | ORC\_z within \(k\); fix α=0.5 |
| Spectral/Laplacian | \(\rho(\widetilde{A}), a(G), \lambda_2(L_{norm})\) | \(A,\widetilde{A}\) | low | density | normalized versions; within-\(k\) residuals |
| Resistance | \(\overline{R}, K_f\) | \(A\) via \(L^+\) | medium | density | \(K_f/n^2\), within-\(k\) z-score |
| Centrality | betweenness/closeness/eigenvector summaries | \(A\) | medium | density, topology class | summaries + within-\(k\) residuals |
| Motifs | triangles, 4-cycles, directed motif counts | \(A\) or \(\widetilde{A}\) | medium | density | normalize by possible motif count; residualize within \(k\) |

## Proxy-only sub-experiments to measure proxy→robustness predictiveness

The goal is to avoid “heavy training for nothing.” This module produces a binary decision: **(a) proxies are predictive enough to justify a TPE-vs-random fight, or (b) they are not, and a pivot is the correct choice.**

### Robustness labels for proxy validation

Let \(p_g(\alpha)\) denote ROC-AUC for graph \(g\) evaluated at perturbation intensity \(\alpha\) (with \(\alpha=0\) as clean). Define:

- Clean performance: \(p_g^{clean}=p_g(0)\).
- Absolute drop curve: \(\Delta_g(\alpha)=p_g^{clean}-p_g(\alpha)\).
- Absolute max drop:  
  \[
  \mathrm{max\_drop}(g)=\max_{\alpha\in \mathcal{A}} \Delta_g(\alpha).
  \]
- Normalized drop curve (robustness-normalized):  
  \[
  RD_g(\alpha)=\frac{p_g^{clean}-p_g(\alpha)}{\max(p_g^{clean}-0.5,\ \epsilon)}.
  \]
- Normalized worst-case degradation:  
  \[
  \mathrm{maxRD}(g)=\max_{\alpha\in\mathcal{A}} RD_g(\alpha).
  \]
- Area under performance curve (AUPC):  
  \[
  \mathrm{AUPC}(g)=\frac{1}{|\mathcal{A}|}\sum_{\alpha\in\mathcal{A}} p_g(\alpha),
  \]
  (or trapezoidal integral if \(\mathcal{A}\) is dense).

**Primary label for proxy tests:** \(y(g)= -\mathrm{maxRD}(g)\) (higher means better robustness), because it controls for clean AUC, which reduces misleading “high-clean drops more” artifacts.

### Predictiveness diagnostics and statistical tests

For each proxy \(P(G)\) and robustness label \(y(G)\):

- **Correlation:** Pearson \(r\) and Spearman \(\rho\), plus **partial correlation controlling for \(k\)** (regress both \(P\) and \(y\) on \(k\), correlate residuals).
- **Mutual information:** continuous MI estimate; evaluate against shuffled baseline with identical marginal distributions.
- **Top-quantile classification:** label “robust” if \(y\) in top 25% and compute AUC of \(P\).
- **Calibration and monotonicity:** isotonic regression fit \(P\to y\); report monotonicity violations and expected calibration error.
- **Stability across subjects:** compute per-subject \(y_s(g)\) and check that the sign of association is consistent across ≥2/3 subjects.

### Lightweight proxy-validation sample design

To balance cost and power:

- **Graph pool size for proxy evaluation:** \(N_{pool}=64\) graphs total (recommended minimum).
- **Regime stratification:** 16 graphs per degree regime (super-sparse / sparse / moderate / near-dense), sampled uniformly over \(p\in[0,1]\). This keeps the pool representative and supports within-regime analysis.
- **Training seeds:** \(S_{pilot}=1\) for proxy validation if budget is tight; \(S_{pilot}=2\) preferred if runtime allows.
- **Subjects:** start with a single subject (e.g., subject 1) for rapid proxy testing; then validate on all 3 subjects only if proxies pass.

**Selection coverage variants to test immediately (proxy-only):**
- none (unconstrained),
- regime (degree-regime stratified only),
- regime_cl_bins_fixed (current; included to quantify “selection constraint washout”).

### Proxy module go/no-go thresholds

A proxy suite is considered viable if **at least one** of the following passes on the pilot labeled set:

- Spearman \(\rho \ge 0.35\) with \(p<0.05\) (after FDR correction across tested proxies), **and**
- AUC for top-25% robust classification \(\ge 0.70\), **and**
- association is directionally consistent in ≥2/3 subjects on a small follow-up check (can be S=1 for the follow-up).

A composite proxy is viable if cross-validated:
- Spearman \(\rho \ge 0.45\) and
- mean absolute error improves ≥15% over a constant baseline predictor.

If no proxy (single or composite) passes, the head-to-head selection experiment is extremely unlikely to show a convincing TPE advantage, and the pivot plan becomes primary.

### Required plots for proxy module

Produce these diagnostics before advancing:

- Scatter: each proxy vs \(y=-\mathrm{maxRD}\), with points colored by regime.
- Heatmap: mean \(y\) over (C,L) tertile bins per regime (to visualize the “basin”).
- Pareto plots: \(p^{clean}\) vs \(\mathrm{maxRD}\) to show tradeoff structure.
- Violin plots: distribution of \(y\) per regime.

Mermaid pipeline for proxy module:

```mermaid
flowchart TD
  A[Generate WS-Flex graphs (stratified by regime)] --> B[Compute graph metrics/proxies]
  B --> C[Train & evaluate pilot robustness labels y = -maxRD]
  C --> D[Proxy diagnostics: corr/MI/AUC + plots]
  D --> E{Proxy viability thresholds met?}
  E -- Yes --> F[Proceed to TPE vs Random decisive experiment]
  E -- No --> G[Activate pivot: basin characterization + practical selection]
```

## Final decisive head-to-head experiment for TPE vs random selection

This module is designed so that, if TPE has any meaningful advantage under your stress test (AR(1) drift), it will show up with high power under mini-scale constraints.

### Core principle: compare under two selection constraints

Run two comparisons:

**Comparison A: regime-stratified (fair, controls degree regime)**  
- Both arms must select exactly 2 graphs per regime (total \(B=8\)).
- No (C,L) bin constraints.

**Comparison B: unconstrained (tests best-of-basin capability)**  
- Both arms select any \(B=8\) graphs.
- This quantifies whether TPE can exploit regime differences and/or bin gradients.

If TPE wins only in unconstrained mode, you can still claim it finds robust graphs, but you must be explicit that it does so partly by reallocating across regimes.

### Parameters for the decisive run

Keep your existing base unless noted:

- dataset: unchanged
- subjects: {1, 3, 4}
- \(B=8\) per arm
- **training seeds: \(S=2\) minimum**, \(S=3\) if feasible
- perturbation: AR(1) drift
- evaluate at **two SNR targets** using the same trained weights:
  - target\_snr\_db = -12 (hard regime)
  - target\_snr\_db = -6 (moderate regime; often less saturated)

Alpha grid:
- keep \(\mathcal{A}=\{0,0.25,0.5,0.75,1\}\) for continuity
- optionally add \(\{0.125,0.375,0.625,0.875\}\) for smoother RD curves if evaluation is cheap

### Selection mechanisms to include

At minimum:

- Random WS-Flex selection (baseline).
- TPE-selected WS-Flex (using chosen proxy objective).

Recommended additions that cost little once graphs are trained:

- Proxy-score selection: choose top-B by a chosen proxy without TPE (isolates whether “search” matters).
- Oracle upper bound: choose top-B by true \(y=-\mathrm{maxRD}\) (quantifies “headroom” in the basin).

### Statistical inference and go/no-go thresholds

Compute per-graph robustness metrics averaged over subjects and training seeds:

- \(p^{clean}(g)\), \(\mathrm{maxRD}(g)\), \(\mathrm{max\_drop}(g)\), \(\mathrm{AUPC}(g)\).
- also report per-graph variance across seeds: \(\mathrm{Var}_{seed}(p^{clean})\), \(\mathrm{Var}_{seed}(\mathrm{maxRD})\).

**Primary comparison metric:** \(\mathrm{maxRD}\) (lower is better).  
**Secondary:** \(p^{clean}\) and Pareto analysis; never interpret robustness without clean.

**Confidence intervals:** hierarchical bootstrap with hierarchy:
- subject → graph → seed (if feasible), else graph → seed with subject fixed.

**Effect size:** Cohen’s \(d\) at the graph level:
\[
d = \frac{\overline{x}_{rand} - \overline{x}_{tpe}}{s_{pooled}},
\]
where \(x=\mathrm{maxRD}\) (or \(\mathrm{max\_drop}\)).

**Decisive GO thresholds (must meet all):**
- \(\Delta = \mathbb{E}[\mathrm{maxRD}_{rand}-\mathrm{maxRD}_{tpe}] \ge 0.05\),
- 95% bootstrap CI for \(\Delta\) excludes 0,
- \(d \ge 0.5\),
- directional consistency: the per-subject mean difference has the same sign in ≥2/3 subjects for both SNR settings (or at least for -6 if -12 is saturated).

**Decisive NO-GO thresholds (pivot triggers):**
- \(|\Delta| < 0.02\) and CI includes 0 in both SNR settings, **or**
- TPE improves robustness only by sacrificing clean AUC by >0.02 (absolute ROC-AUC) in regime-stratified mode, indicating a tradeoff rather than true robustness gain.

### Recommended visualizations for the decisive run

- Violin/box: \(\mathrm{maxRD}\) per arm (graph-level points overlaid).
- Pareto front: \(p^{clean}\) vs \(\mathrm{maxRD}\), with convex hull / Pareto set.
- RD curves: median + 25/75% band of \(RD(\alpha)\) per arm.
- Waterfall: per-graph \(\mathrm{maxRD}\) sorted, comparing arms (high diagnostic value).
- Heatmap overlay: where selected graphs fall in (C,L) bins for each arm (reveals whether selection differences are real or constrained away).

## Robust pivot plan if TPE does not outperform random

If the decisive module ends in NO-GO, the strongest version of the pivot is to shift Plot 2 from “method wins” to “**topology basin characterization with practical selection guidance**.”

A robust pivot contribution has three pillars:

**Basin existence and density**  
Show that within WS-Flex (under capacity control), robust graphs occur at high frequency. This is consistent with design-space coverage perspectives that emphasize structural regions (“sweet spots”) rather than isolated optima. [R2] Plot: heatmaps of mean robustness over (C,L) bins and regimes.

**Family effect dominates search effect**  
Demonstrate that WS-Flex constrained graphs (random inside basin) beat “external random” or other unconstrained wiring families under matched capacity/protocol. This makes Plot 2 a topology-family result rather than a NAS result. This aligns with robustness-architecture work showing structure measures can stratify robustness and that design-space matters. [R1]

**Actionable selection rule**  
Provide one or two cheap, training-light selection recipes that match TPE performance:
- regime-stratified random + light proxy filtering (e.g., remove the lowest sigma quartile within each regime),
- or direct proxy-score selection using a validated composite proxy (without TPE).

This pivot can still be positioned as “training-free or training-light robustness engineering,” emphasizing:
- a reproducible topology search space,
- validated structural correlates of robustness,
- and an explicit absence (or small size) of gains from sophisticated selection.

This is more defensible than forcing a weak “TPE wins” claim.

## Source notes for metric choices and framing

[R1] Robustness-architecture work explicitly defines (i) spectral-radius-based entropy \(H=\log(\rho(A))\) and (ii) ORC via Wasserstein distance between neighborhood measures, presenting both as pre-training structural correlates of robustness, and uses standard clustering definitions. (Communications Engineering, 2022; “Exploring robust architectures for deep artificial neural networks,” Methods section where entropy and ORC are defined.)

[R2] Relational-graph architecture work motivates WS(-flex) style graph generators, explores a design space spanned by clustering coefficient and average path length, and reports performance as approximately a smooth function over that space with an identifiable “sweet spot.” (ICML 2020 paper “Graph Structure of Neural Networks,” abstract/overview content.)

[R3] Small-worldness sigma is commonly defined as \(\sigma=(C/C_{rand})/(L/L_{rand})\) in small-world benchmarking literature (e.g., the sigma formulation used historically to quantify “small-worldness” beyond having high clustering and low path length).

## Final experiment specification

```text
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
```