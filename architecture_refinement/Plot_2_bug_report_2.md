====================================================================
BUG SUMMARY: Plot 2 Correlated Perturbation Not Applied (Gaussian Fallback)
====================================================================

Status
------
CRITICAL BUG – invalidates current Plot 2 correlated-perturbation results.
The plotted results labeled as "ar1_drift AUPC" are almost certainly
generated using the Gaussian corruption pipeline.

--------------------------------------------------------------------
Observed Symptoms
--------------------------------------------------------------------

1) Plot 2 results show near-identical AUPC across:
   - Random WS-Flex
   - NAS-selected WS-Flex
   - NCP baseline

   This mirrors earlier Gaussian-noise failures and contradicts the
   escalated pilot results for AR(1) drift.

2) report.txt explicitly references Gaussian parameters:
   - "sigma_max (gaussian): 100.0"
   - No mention of AR(1) parameters (rho, alpha_max, target_snr_db)

3) per_seed_aupc.csv and per_graph_aupc.csv include:
   - sigma_max column
   - no perturbation_type column
   - no AR(1)-specific fields

4) Plot title claims "ar1_drift AUPC" while the underlying artifacts
   still reflect Gaussian execution.

5) Earlier pilot experiments demonstrate that AR(1) drift at escalated
   severity produces a large, monotonic ROC-AUC drop (~0.11), making
   it implausible that AR(1) is genuinely active in this run.

--------------------------------------------------------------------
Intended Behavior
--------------------------------------------------------------------

For Plot 2 with correlated perturbations:

- The runner must apply the specified perturbation_type exactly:
    perturbation_type ∈ {ar1_drift, spatial_gaussian, emg_band}

- Perturbation intensity must be controlled via:
    - target_snr_db
    - computed alpha_max
    - alpha_grid ∈ {0, .25, .5, .75, 1.0} * alpha_max

- Gaussian-specific parameters (e.g., sigma_max) must not be used or
  logged for non-Gaussian perturbations.

- All outputs must explicitly record:
    - perturbation_type
    - target_snr_db
    - empirical_snr_db
    - perturbation parameters (rho, ell, band, envelope)

- Plot labels must be derived from the run metadata, not hardcoded.

--------------------------------------------------------------------
Likely Causes
--------------------------------------------------------------------

Primary causes (most likely):
1) Perturbation selection logic silently defaults to Gaussian when
   perturbation_type is missing, misspelled, or not threaded into the
   corruption function.

2) The corruption pipeline still keys off 'sigma_max' regardless of
   perturbation_type, causing Gaussian noise to be applied even when
   AR(1) is requested.

3) report.txt generation uses a Gaussian-only template and is unaware
   of correlated perturbation modes.

Secondary causes:
4) Metrics aggregation reads from a Gaussian-corrupted column even if
   AR(1) corruption is applied elsewhere.

5) Plotting script labels the figure as "ar1_drift" based on config
   rather than on the executed perturbation recorded in the artifacts.

--------------------------------------------------------------------
Required Fixes
--------------------------------------------------------------------

- Add hard runtime assertions:
    If perturbation_type == "ar1_drift", assert that rho exists and
    sigma_max is not used.

- Add mandatory metadata fields to all output tables:
    perturbation_type, target_snr_db, empirical_snr_db, params

- Remove Gaussian-specific parameters from non-Gaussian runs.

- Compute and log a simple diagnostic fingerprint:
    lag-1 autocorrelation of injected noise (≈ rho for AR(1), ≈ 0 for Gaussian)

- Make plotting scripts read perturbation_type from run metadata.

--------------------------------------------------------------------
Conclusion
--------------------------------------------------------------------

The current Plot 2 results do not reflect a failure of the NAS method or
of correlated perturbations. They reflect a configuration/plumbing bug
that causes Gaussian corruption to be applied and logged even when
correlated perturbations are intended.

Until this bug is fixed and verified, Plot 2 conclusions are invalid.

--------------------------------------------------------------------
Fixes applied (implementation)
--------------------------------------------------------------------

- Runtime assertions (unified_experiment_runner.py):
  - For perturbation_type == "ar1_drift", assert test_perturb_ar1_rho in (0, 1).
  - Assert ar1_drift uses the correlated path (data-derived alpha_max), not gaussian sigma_max.
  - Assert the created augmentor's noise_type is ar1_drift when using the correlated path.

- Mandatory metadata in outputs:
  - Result rows now include perturbation_type (alias for noise_type) and params (e.g. {"rho": ...} for ar1_drift).
  - Analyzer writes perturbation_type, target_snr_db, empirical_snr_db in per_seed_aupc.csv, per_graph_aupc.csv, and bootstrap_diff.json (primary_perturbation_type).
  - Report uses "alpha_max (type, data-derived)" for correlated primary; "sigma_max (type)" for gaussian.

- Report/plot must reflect actual applied perturbation:
  - Analyzer validates manifest primary_perturbation_type against data: if data contain only gaussian but manifest says ar1_drift, it warns and uses gaussian for report/plot.
  - Backward compatibility: when manifest lacks perturbation_types, report includes "Manifest missing perturbation_types; defaulting to gaussian."

- Skip-check (experiment_utils.py): for correlated types, intensity completeness uses count match only (not value match), since actual intensities are data-derived (alpha * alpha_max).

- Optional diagnostic: unified runner logs lag-1 autocorrelation of injected noise at max intensity (expected ~rho for AR(1), ~0 for gaussian).

====================================================================
