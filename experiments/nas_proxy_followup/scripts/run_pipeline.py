#!/usr/bin/env python3
"""
End-to-end NAS proxy follow-up pipeline (NAS_Proxy_followup_spec.md §6).

Runs steps 01→11 with merged YAML configs, stdout logging, and go/no-go gates
after the realization audit (spec §4B.6).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config_util import load_merged_configs, resolve_path


class GateResult(str, Enum):
    GO = "GO"
    WARN = "WARN"
    NO_GO = "NO_GO"


@dataclass
class RealizationGateDecision:
    result: GateResult
    recommended_mapping_scheme: str
    messages: List[str]


def _log(msg: str, *, box: bool = False) -> None:
    if box:
        line = "=" * 72
        print(f"\n{line}\n{msg}\n{line}\n", flush=True)
    else:
        print(f"[nas_proxy_pipeline] {msg}", flush=True)


def _run_script(
    repo_root: Path,
    script_rel: str,
    config_paths: Sequence[Path],
    extra_args: Optional[List[str]] = None,
) -> None:
    script = repo_root / script_rel
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    cmd = [sys.executable, str(script)]
    for c in config_paths:
        cmd.extend(["--config", str(c)])
    if extra_args:
        cmd.extend(extra_args)
    _log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_realization_audit_gate(
    summary_path: Path,
    *,
    min_deterministic_hamming: float = 0.02,
    diversity_improvement_ratio: float = 1.25,
) -> RealizationGateDecision:
    """
    Spec §4B.6: if deterministic realization collapses diversity vs stochastic schemes,
    recommend a diversity-preserving mapping for probe/downstream.

    Uses mean pairwise Hamming on realized hidden masks (higher = more diverse).
    """
    messages: List[str] = []
    if not summary_path.exists():
        return RealizationGateDecision(
            GateResult.WARN,
            "deterministic_baseline",
            [f"Summary not found at {summary_path}; cannot evaluate gate. Proceed with caution."],
        )

    data = _load_json(summary_path)
    by_scheme = data.get("by_scheme") or {}
    raw_mean = data.get("raw_graph_dist_mean")

    det = by_scheme.get("deterministic_baseline") or {}
    rand = by_scheme.get("random_io_anchors") or {}
    degw = by_scheme.get("degree_weighted_io_anchors") or {}

    det_h = det.get("mean_hamming_pdist")
    rand_h = rand.get("mean_hamming_pdist")
    degw_h = degw.get("mean_hamming_pdist")

    messages.append(
        f"Realization diversity (mean pairwise Hamming on hidden masks): "
        f"deterministic={det_h}, random_io={rand_h}, degree_weighted_io={degw_h}"
    )
    if raw_mean is not None:
        messages.append(f"Raw graph adjacency distance (mean Hamming): {raw_mean}")

    stochastic_refs: List[Tuple[str, Optional[float]]] = [
        ("random_io_anchors", rand_h),
        ("degree_weighted_io_anchors", degw_h),
    ]
    best_stoch_name = "random_io_anchors"
    best_stoch_h: Optional[float] = None
    for name, val in stochastic_refs:
        if val is not None and (best_stoch_h is None or val > best_stoch_h):
            best_stoch_h = val
            best_stoch_name = name

    recommended = "deterministic_baseline"

    if det_h is None:
        messages.append("Gate: WARN — no deterministic diversity metric (too few masks?).")
        return RealizationGateDecision(GateResult.WARN, recommended, messages)

    if (
        det_h < min_deterministic_hamming
        and best_stoch_h is not None
        and best_stoch_h >= diversity_improvement_ratio * max(det_h, 1e-12)
    ):
        recommended = best_stoch_name
        messages.append(
            f"Gate: NO_GO — deterministic mean Hamming ({det_h:.4f}) < {min_deterministic_hamming} "
            f"and {best_stoch_name} is ≥{diversity_improvement_ratio}× higher → likely diversity collapse."
        )
        messages.append(
            f"Recommendation: use mapping_scheme={recommended} for probe/downstream "
            "(update configs: mapping_schemes / downstream mapping list)."
        )
        return RealizationGateDecision(GateResult.NO_GO, recommended, messages)

    if det_h < min_deterministic_hamming:
        messages.append(
            f"Gate: WARN — deterministic mean Hamming ({det_h:.4f}) below {min_deterministic_hamming}; "
            "consider stochastic mapping if proxies look degenerate."
        )
        return RealizationGateDecision(GateResult.WARN, recommended, messages)

    messages.append(
        f"Gate: GO — deterministic diversity metric OK (≥ {min_deterministic_hamming})."
    )
    return RealizationGateDecision(GateResult.GO, recommended, messages)


def default_config_stack(*, full: bool) -> List[Path]:
    """Default YAML stack: full includes downstream + perturbation."""
    base = _REPO_ROOT / "experiments/nas_proxy_followup/configs"
    paths: List[Path] = [
        base / "base.yaml",
        base / "topology/wsflex_panel.yaml",
        base / "topology/wsflex_mapping_ablation.yaml",
        base / "dataset/bnci2014_001_cross_session.yaml",
        base / "probe/probe_3epoch.yaml",
    ]
    if full:
        paths.append(base / "downstream/downstream_full.yaml")
        paths.append(base / "perturbation/gaussian_local.yaml")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=None,
        metavar="PATH",
        help="YAML config (repeat; later overrides). If omitted, defaults are used.",
    )
    parser.add_argument(
        "--topology-limit",
        type=int,
        default=0,
        help="Passed to 01 (--limit). 0 = full panel.",
    )
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=0,
        help="Passed to 04 (--limit). 0 = all probe manifest rows.",
    )
    parser.add_argument(
        "--downstream-limit",
        type=int,
        default=0,
        help="Passed to 07 (--limit). 0 = all rows.",
    )
    parser.add_argument(
        "--perturbation-limit",
        type=int,
        default=0,
        help="Passed to 08 (--limit). 0 = all rows.",
    )
    parser.add_argument(
        "--skip-downstream",
        action="store_true",
        help="Stop after step 05 (skip downstream, perturbation, merge, validation, screening).",
    )
    parser.add_argument(
        "--skip-perturbation",
        action="store_true",
        help="Skip step 08 only (still runs downstream, merge, validation, screening).",
    )
    parser.add_argument(
        "--no-stop-on-gate-failure",
        action="store_true",
        help="Continue after step 02 even if realization gate is NO_GO (default: exit 2).",
    )
    parser.add_argument(
        "--min-deterministic-hamming",
        type=float,
        default=0.02,
        help="Gate threshold on mean pairwise Hamming for deterministic_baseline masks.",
    )
    parser.add_argument(
        "--diversity-improvement-ratio",
        type=float,
        default=1.25,
        help="Stochastic scheme must exceed deterministic by this factor to trigger NO_GO.",
    )
    args = parser.parse_args()

    repo_root = _REPO_ROOT

    if args.configs:
        config_paths = [Path(p).resolve() for p in args.configs]
    else:
        config_paths = default_config_stack(full=not args.skip_downstream)

    cfg = load_merged_configs(config_paths)

    _log(
        "NAS Proxy Follow-up — end-to-end pipeline\n"
        f"Repo root: {repo_root}\n"
        f"Configs ({len(config_paths)}): "
        + ", ".join(p.name for p in config_paths),
        box=True,
    )

    extra_01: List[str] = []
    if args.topology_limit and args.topology_limit > 0:
        extra_01.extend(["--limit", str(args.topology_limit)])

    # Step 1 — same merged stack so 01 sees topology/wsflex_panel keys
    _log("STEP 1/11 — Build topology panel (01_build_topology_panel.py)")
    _run_script(
        repo_root,
        "experiments/nas_proxy_followup/scripts/01_build_topology_panel.py",
        config_paths,
        extra_args=extra_01 if extra_01 else None,
    )
    topo_csv = resolve_path(cfg, "output_manifest", "experiments/nas_proxy_followup/manifests/topology_panel.csv")
    if not topo_csv.exists():
        _log(f"ERROR: Expected manifest missing: {topo_csv}")
        sys.exit(1)
    with open(topo_csv, encoding="utf-8") as f:
        n_topo = max(0, sum(1 for _ in f) - 1)
    _log(f"OK — topology_panel.csv rows: {n_topo}")

    # Step 2
    _log("STEP 2/11 — Realization diversity audit (02_analyze_realization_diversity.py)")
    _run_script(
        repo_root,
        "experiments/nas_proxy_followup/scripts/02_analyze_realization_diversity.py",
        config_paths,
    )

    out_real = resolve_path(
        cfg,
        "realization_output_dir",
        "experiments/nas_proxy_followup/outputs/realization_analysis",
    )
    summary_path = out_real / "realization_diversity_summary.json"

    decision = evaluate_realization_audit_gate(
        summary_path,
        min_deterministic_hamming=args.min_deterministic_hamming,
        diversity_improvement_ratio=args.diversity_improvement_ratio,
    )
    for m in decision.messages:
        _log(m)
    _log(f"Gate result: {decision.result.value} | recommended mapping_scheme: {decision.recommended_mapping_scheme}")

    if decision.result == GateResult.NO_GO and not args.no_stop_on_gate_failure:
        _log(
            "HALT — Realization audit NO_GO. "
            "Adjust mapping schemes or pass --no-stop-on-gate-failure to continue.",
            box=True,
        )
        sys.exit(2)
    if decision.result == GateResult.WARN:
        _log("Continuing with WARN (review diversity metrics before trusting probe outcomes).")

    # Steps 3–5
    _log("STEP 3/11 — Build probe manifest (03_build_probe_manifest.py)")
    _run_script(repo_root, "experiments/nas_proxy_followup/scripts/03_build_probe_manifest.py", config_paths)

    extra_04: List[str] = []
    if args.probe_limit and args.probe_limit > 0:
        extra_04.extend(["--limit", str(args.probe_limit)])

    _log("STEP 4/11 — Probe training + proxies (04_run_probe_stage.py)")
    _run_script(
        repo_root,
        "experiments/nas_proxy_followup/scripts/04_run_probe_stage.py",
        config_paths,
        extra_args=extra_04 if extra_04 else None,
    )

    _log("STEP 5/11 — Aggregate probe metrics (05_aggregate_probe_metrics.py)")
    _run_script(repo_root, "experiments/nas_proxy_followup/scripts/05_aggregate_probe_metrics.py", config_paths)

    agg_csv = resolve_path(cfg, "probe_output_dir", "experiments/nas_proxy_followup/outputs/probe_runs") / "probe_metrics_aggregated.csv"
    if agg_csv.exists():
        with open(agg_csv, encoding="utf-8") as f:
            n_agg = max(0, sum(1 for _ in f) - 1)
        _log(f"OK — probe_metrics_aggregated.csv rows: {n_agg}")
    else:
        _log(f"WARN — probe aggregation CSV not found at {agg_csv}")

    if args.skip_downstream:
        _log("STOP — Skipping steps 6–11 (--skip-downstream). Probe path complete.", box=True)
        return

    if not any("downstream" in p.name for p in config_paths):
        _log("ERROR: Downstream configs missing from stack. Add downstream/downstream_full.yaml or use default configs.")
        sys.exit(1)

    extra_07: List[str] = []
    if args.downstream_limit and args.downstream_limit > 0:
        extra_07.extend(["--limit", str(args.downstream_limit)])

    _log("STEP 6/11 — Downstream manifest (06_build_downstream_manifest.py)")
    _run_script(repo_root, "experiments/nas_proxy_followup/scripts/06_build_downstream_manifest.py", config_paths)

    _log("STEP 7/11 — Downstream training (07_run_downstream_stage.py) [long-running]")
    _run_script(
        repo_root,
        "experiments/nas_proxy_followup/scripts/07_run_downstream_stage.py",
        config_paths,
        extra_args=extra_07 if extra_07 else None,
    )

    if not args.skip_perturbation:
        extra_08: List[str] = []
        if args.perturbation_limit and args.perturbation_limit > 0:
            extra_08.extend(["--limit", str(args.perturbation_limit)])
        _log("STEP 8/11 — Perturbation evaluation (08_evaluate_perturbation.py)")
        _run_script(
            repo_root,
            "experiments/nas_proxy_followup/scripts/08_evaluate_perturbation.py",
            config_paths,
            extra_args=extra_08 if extra_08 else None,
        )
    else:
        _log("SKIP — Step 8 perturbation (--skip-perturbation)")

    _log("STEP 9/11 — Merge results (09_merge_results.py)")
    _run_script(repo_root, "experiments/nas_proxy_followup/scripts/09_merge_results.py", config_paths)

    _log("STEP 10/11 — Validate proxies (10_validate_proxies.py)")
    _run_script(repo_root, "experiments/nas_proxy_followup/scripts/10_validate_proxies.py", config_paths)

    _log("STEP 11/11 — Screening simulation (11_screening_simulation.py)")
    _run_script(repo_root, "experiments/nas_proxy_followup/scripts/11_screening_simulation.py", config_paths)

    _log("Pipeline finished successfully.", box=True)


if __name__ == "__main__":
    main()
