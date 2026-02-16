Below is a clean, implementation-ready **design specification** for the proposed **CfC-as-Carry-Gate** upgrade, written as a standalone markdown document. You can drop this directly into your repo as `HYDRA_V3_Spec.md`.

---

# HYDRA_V3_Spec.md

## HYDRA-V3: CfC-Controlled Adaptive Carry Gate

### Status

**Proposed architectural revision** to HYDRA_v2
**Motivation**: Improve worst-case robustness and theoretical defensibility by replacing fragile explicit adaptive gating with a recurrent, continuous-time controller.

---

## 1. Motivation and Design Rationale

HYDRA_v2 employs explicit adaptive skip / carry mechanisms outside the recurrent core to regulate information flow under noise and perturbation. Empirical ablations indicate that while these gates do not significantly affect mean robustness metrics (e.g., AUPC), they can **harm worst-case relative degradation**, suggesting sensitivity to corrupted or ill-conditioned statistics.

HYDRA_v3 addresses this limitation by:

* **Eliminating explicit feedforward carry gates**, and
* **Subsuming adaptive routing into the recurrent dynamics** via a **small Closed-Form Continuous-time (CfC) controller**.

This approach:

* Preserves the functional role of adaptive carry/skip routing,
* Introduces temporal inertia and smoothing into the gating decision,
* Avoids reliance on brittle instantaneous statistics (e.g., variance, SNR),
* Aligns adaptive control with CfC’s bounded, continuous-time dynamics.

---

## 2. High-Level Architectural Change

### HYDRA_v2 (simplified)

```
Feature Extractor
        ↓
  Explicit Adaptive Gate (MLP / stat-driven)
        ↓
   Residual / Carry Routing
        ↓
   Recurrent Bin (CfC / Wired CfC)
```

### HYDRA_v3 (proposed)

```
Feature Extractor
        ↓
   Recurrent Bin (CfC / Wired CfC)
        ↓
  CfC Carry Controller (new, internal)
        ↓
   Implicit Adaptive Mixing (inside recurrence)
```

**Key change**: Adaptive routing is no longer an external residual path; it is computed internally by a recurrent controller with its own state.

---

## 3. CfC-as-Carry-Gate Design

### 3.1 Controller Overview

For each temporal bin ( b ), introduce a **low-dimensional CfC controller state**:

[
c_t^{(b)} \in \mathbb{R}^{d_c}, \quad d_c \in {1,2}
]

This controller:

* Receives the same bin-level inputs as the main recurrent unit (or a reduced subset),
* Evolves via CfC dynamics,
* Outputs a **carry coefficient** ( \alpha_t^{(b)} \in (0,1) ).

---

### 3.2 State Updates

#### Main recurrent state (existing)

[
\tilde{h}*t^{(b)} = \mathrm{CfC}*{\text{main}}!\left(h_{t-1}^{(b)},, z_t^{(b)},, \Delta t\right)
]

#### Carry controller state (new)

[
c_t^{(b)} = \mathrm{CfC}*{\text{gate}}!\left(c*{t-1}^{(b)},, z_t^{(b)},, \Delta t\right)
]

---

### 3.3 Adaptive Mixing Rule

The carry coefficient is computed as:
[
\alpha_t^{(b)} = \sigma!\left(W_c, c_t^{(b)} + b_c\right)
]

The final recurrent update becomes:
[
h_t^{(b)} = \alpha_t^{(b)}, h_{t-1}^{(b)} ;+; \bigl(1-\alpha_t^{(b)}\bigr), \tilde{h}_t^{(b)}
]

This reproduces the functional role of an adaptive skip/carry gate **without any explicit residual routing outside the recurrent compartment**.

---

## 4. Architectural Properties

### 4.1 Functional Equivalence

* Recovers learned carry/skip behavior from HYDRA_v2.
* Can emulate static gates, dynamic gates, or pure recurrence as special cases.

### 4.2 Improved Robustness Characteristics

* Gating decisions are **temporally smoothed**.
* No dependence on instantaneous variance or SNR estimates.
* Reduced sensitivity to single-step corruption spikes.

### 4.3 Parameter Efficiency

* Adds only ( \mathcal{O}(d_c) ) parameters per bin.
* No additional convolutional or dense layers required.

---

## 5. Stability and Theoretical Framing

HYDRA_v3 enables a **more defensible stability narrative**:

* Adaptive control is governed by CfC dynamics with bounded nonlinearities.
* Carry coefficients are:

  * Bounded (( \alpha_t \in (0,1) )),
  * Continuous in time,
  * State-dependent rather than stat-dependent.
* This supports claims of:

  * Bounded-input bounded-state (BIBS-style) behavior,
  * Reduced local sensitivity under perturbation.

**Important note**:
HYDRA_v3 does **not** claim global end-to-end Lipschitz guarantees. Instead, it localizes adaptive routing within a recurrent mechanism that admits boundedness analysis under standard assumptions.

---

## 6. Implementation Notes

### 6.1 Defaults

* Controller dimension: `d_c = 1`
* Shared or per-bin controller: **per-bin recommended**
* Nonlinearity: `sigmoid` (hard-sigmoid optional for stability)
* Initialization: bias ( b_c \approx 0 ) (neutral carry at start)

### 6.2 Optional Regularization

* Penalize high temporal variation in ( \alpha_t ):
  [
  \mathcal{L}*{\text{smooth}} = \lambda \sum_t |\alpha_t - \alpha*{t-1}|
  ]
* Optional spectral norm constraints on ( W_c )

---

## 7. Expected Empirical Outcomes

Compared to HYDRA_v2:

* **No regression in baseline accuracy**
* **Recovery of AUPC performance**
* **Improved worst-case RD under high perturbation**
* Reduced failure modes in SSVEP / ERP regimes
* Cleaner ablation story: adaptive control is intrinsic, not auxiliary

---

## 8. Evaluation Checklist (HYDRA_v3 Acceptance Criteria)

* [ ] Mean AUPC ≥ HYDRA_v2
* [ ] Worst-case RD ≤ HYDRA_v2
* [ ] Reduced variance of ( \alpha_t ) under noise
* [ ] No reliance on explicit statistical gates
* [ ] Ablation: removing controller reverts to HYDRA_v1-like behavior

---

## 9. Summary

HYDRA_v3 replaces brittle explicit adaptive gates with a **CfC-driven carry controller**, embedding adaptive routing into the recurrent dynamics themselves. This yields a model that is:

* More robust in failure regimes,
* Architecturally simpler,
* Better aligned with continuous-time stability arguments,
* And empirically motivated by HYDRA_v2 ablation results.

---

**End of Specification**
