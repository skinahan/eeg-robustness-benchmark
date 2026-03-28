# NAS Proxy Follow-up

Follow-up experiment pipeline for WS-Flex topologies, CfC realization audit, probe-run proxies (Jacobian / finite-difference), downstream training, and screening analysis.

## End-to-end pipeline script

Use the project Conda environment (MOABB / Torch), e.g.:

```bash
conda activate ncp_robustness_proj
```

Run all steps (01–11) with merged configs, step banners, and a **go/no-go gate** after the realization audit (spec §4B.6):

```bash
python experiments/nas_proxy_followup/scripts/run_pipeline.py
```

**EEG trial shape:** MOABB returns trials as `(N, n_channels, n_times)`. The probe model uses `src/probe/eeg_layout.py` (`infer_eeg_n_channels`) so the encoder `Linear` matches the channel dimension (not the time length).

**Labels:** `encode_moabb_labels` supports string or integer class codes; `n_outputs` defaults to `max(y_encoded)+1` when not set in YAML.

**Tests:** `tests/test_nas_proxy_eeg_layout.py` is Torch-free. `tests/test_nas_proxy_followup.py` imports wiring (requires PyTorch); run with the same Conda env as training.

- **Probe-only** (stops after step 05): `--skip-downstream`
- **Smaller topology panel**: `--topology-limit N`
- **Gate exits with code 2** if deterministic realization looks collapsed vs stochastic schemes; override with `--no-stop-on-gate-failure`
- **Tune gate thresholds**: `--min-deterministic-hamming`, `--diversity-improvement-ratio`

Custom YAML stack (later files override):

```bash
python experiments/nas_proxy_followup/scripts/run_pipeline.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/topology/wsflex_panel.yaml \
  ...
```

## Objective

Validate whether probe-run dynamical proxies improve topology screening versus static graph metrics (TE, ORC, density), after auditing whether WS-Flex → CfC realization preserves structural diversity.

## Dataset and protocol

Default: **BNCI2014_001**, **cross_session** (train `0train`, validation `1test`). Configurable in `configs/dataset/bnci2014_001_cross_session.yaml`.

## Topology panel generation

Builds stratified WS-Flex graphs over `(k, p)` and writes `manifests/topology_panel.csv` plus `.npz` adjacency files.

```bash
python experiments/nas_proxy_followup/scripts/01_build_topology_panel.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/topology/wsflex_panel.yaml
```

Optional: `--limit N` for a small panel.

## Realization audit

Compares mapping schemes (`deterministic_baseline`, `random_io_anchors`, `degree_weighted_io_anchors`) and writes realization manifests and diversity summaries.

```bash
python experiments/nas_proxy_followup/scripts/02_analyze_realization_diversity.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/topology/wsflex_mapping_ablation.yaml
```

**Decision:** If distinct raw graphs map to near-duplicate realized masks under the deterministic scheme, prefer a stochastic scheme for later stages (see spec §4B.6).

## Probe stage

1. Build probe indices and manifest:

```bash
python experiments/nas_proxy_followup/scripts/03_build_probe_manifest.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/dataset/bnci2014_001_cross_session.yaml \
  --config experiments/nas_proxy_followup/configs/probe/probe_3epoch.yaml
```

2. Run probe training + proxy extraction:

```bash
python experiments/nas_proxy_followup/scripts/04_run_probe_stage.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/dataset/bnci2014_001_cross_session.yaml \
  --config experiments/nas_proxy_followup/configs/probe/probe_3epoch.yaml
```

3. Aggregate probe metrics:

```bash
python experiments/nas_proxy_followup/scripts/05_aggregate_probe_metrics.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/probe/probe_3epoch.yaml
```

## Downstream stage

```bash
python experiments/nas_proxy_followup/scripts/06_build_downstream_manifest.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/downstream/downstream_full.yaml

python experiments/nas_proxy_followup/scripts/07_run_downstream_stage.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/dataset/bnci2014_001_cross_session.yaml \
  --config experiments/nas_proxy_followup/configs/downstream/downstream_full.yaml
```

## Perturbation evaluation

```bash
python experiments/nas_proxy_followup/scripts/08_evaluate_perturbation.py \
  --config experiments/nas_proxy_followup/configs/base.yaml \
  --config experiments/nas_proxy_followup/configs/perturbation/gaussian_local.yaml
```

## Proxy validation and screening

```bash
python experiments/nas_proxy_followup/scripts/09_merge_results.py \
  --config experiments/nas_proxy_followup/configs/base.yaml

python experiments/nas_proxy_followup/scripts/10_validate_proxies.py \
  --config experiments/nas_proxy_followup/configs/base.yaml

python experiments/nas_proxy_followup/scripts/11_screening_simulation.py \
  --config experiments/nas_proxy_followup/configs/base.yaml
```

## Key manifests and outputs

| Artifact | Location |
|----------|----------|
| Topology panel | `manifests/topology_panel.csv` |
| Realization audit | `manifests/topology_panel_realization.csv`, `outputs/realization_analysis/` |
| Probe manifest | `manifests/probe_run_manifest.csv` |
| Probe metrics | `outputs/probe_runs/probe_metrics_aggregated.csv` |
| Merged table | `outputs/analysis/merged_run_table.csv` |

## Reproduction

Run scripts **01 → 02** before large probe/downstream jobs. All paths in YAML are relative to the **repository root** unless overridden by `repo_root` in `configs/base.yaml`.

## Implementation notes

- **CfC model:** Paper 3–style `CfCProbeModel` (encoder → wired CfC → readout) in `src/probe/probe_runner.py`.
- **Wiring schemes:** Implemented in `architecture_refinement/arbitrary_wiring.py` (`random_io`, `degree_weighted_io` strategies) and mapped in `src/wiring/cfc_realizer.py`.
