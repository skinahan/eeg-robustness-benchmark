"""
Phase 6: Pipeline integrity checks (Plot_2_Investigation.txt).

Run when sensitivity is still missing after Phase 5. Verifies:
1) Perturbation active and logged (Patch 0.2 fingerprint)
2) AUPC/computation reads correct columns and perturbation_type
3) Masked-weight wiring: effective sparsity stats per graph (optional)
4) Selection pool: no duplicate graph IDs, disjoint draws if intended
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def check_fingerprint(plot2_dir: Path, strict_spec: bool = False) -> Dict[str, Any]:
    """1) Confirm perturbation fingerprint exists and has expected keys. Spec §7: for AR(1), lag1 >= 0.90 when strict_spec."""
    fp_path = plot2_dir / "diagnostics" / "perturbation_fingerprint.json"
    out = {"path": str(fp_path), "exists": fp_path.exists(), "pass": False, "details": {}}
    if not fp_path.exists():
        out["message"] = "PATCH 0.2 fingerprint not found; run with --plot2_diagnostics_dir to generate."
        return out
    try:
        data = json.loads(fp_path.read_text(encoding="utf-8"))
        out["details"] = data
        has_lag1 = "lag1_autocorrelation" in data
        has_residual = "residual_mean" in data and "residual_std" in data
        out["pass"] = has_lag1 or has_residual
        # Spec §7 hard gate: for AR(1) runs, lag1_autocorrelation must be >= 0.90
        if strict_spec and out["pass"]:
            pt = data.get("perturbation_type")
            if str(pt).strip().lower() in ("ar1_drift", "ar1"):
                lag1 = data.get("lag1_autocorrelation")
                try:
                    lag1_f = float(lag1) if lag1 is not None else float("nan")
                except (TypeError, ValueError):
                    lag1_f = float("nan")
                if not (np.isfinite(lag1_f) and lag1_f >= 0.90):
                    out["pass"] = False
                    out["message"] = f"Spec §7: AR(1) requires lag1_autocorrelation >= 0.90, got {lag1!r}"
                else:
                    out["message"] = "Fingerprint present; lag1 >= 0.90 for AR(1)."
            else:
                out["message"] = "Fingerprint present; verify lag1 ~ rho for AR(1)." if out["pass"] else "Fingerprint missing expected keys."
        else:
            out["message"] = "Fingerprint present; verify lag1 ~ rho for AR(1)." if out["pass"] else "Fingerprint missing expected keys."
    except Exception as e:
        out["message"] = str(e)
    return out


def check_manifest_primary_perturbation(plot2_dir: Path) -> Dict[str, Any]:
    """2a) Manifest primary_perturbation_type and result columns."""
    manifest_path = plot2_dir / "plot2_manifest.json"
    out = {"path": str(manifest_path), "exists": manifest_path.exists(), "pass": False, "primary_perturbation_type": None}
    if not manifest_path.exists():
        out["message"] = "Manifest not found."
        return out
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        out["primary_perturbation_type"] = m.get("primary_perturbation_type") or m.get("perturbation_type")
        out["perturbation_types"] = m.get("perturbation_types")
        out["pass"] = bool(out["primary_perturbation_type"])
        out["message"] = "Manifest has primary perturbation type." if out["pass"] else "Manifest missing primary_perturbation_type."
    except Exception as e:
        out["message"] = str(e)
    return out


def check_collapse_scores(plot2_dir: Path) -> Dict[str, Any]:
    """Spec §7: selection_collapse_scores must be <= 0.50 for each WS-Flex method."""
    manifest_path = plot2_dir / "plot2_manifest.json"
    out = {"path": str(manifest_path), "exists": manifest_path.exists(), "pass": True, "scores": {}, "message": ""}
    if not manifest_path.exists():
        out["pass"] = False
        out["message"] = "Manifest not found."
        return out
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        scores = m.get("selection_collapse_scores") or {}
        out["scores"] = scores
        for method, val in scores.items():
            if val is not None and np.isfinite(float(val)) and float(val) > 0.50:
                out["pass"] = False
                out["message"] = f"Spec §7: collapse_score for {method!r} is {val} > 0.50."
                return out
        out["message"] = "All selection_collapse_scores <= 0.50." if scores else "No selection_collapse_scores in manifest (run may predate spec)."
    except Exception as e:
        out["pass"] = False
        out["message"] = str(e)
    return out


def check_max_drop_in_outputs(plot2_dir: Path) -> Dict[str, Any]:
    """Spec §7: analysis outputs must include max_drop."""
    analysis_dir = plot2_dir / "analysis"
    per_seed_path = analysis_dir / "per_seed_aupc.csv"
    out = {"path": str(per_seed_path), "exists": per_seed_path.exists(), "pass": True, "message": ""}
    if not per_seed_path.exists():
        out["pass"] = True  # Analysis not run yet; skip
        out["message"] = "Analysis not run (per_seed_aupc.csv missing); skip max_drop check."
        return out
    try:
        import pandas as pd
        df = pd.read_csv(per_seed_path, nrows=1)
        if "max_drop" not in df.columns:
            out["pass"] = False
            out["message"] = "Spec §7: per_seed metrics must include max_drop column."
        else:
            out["message"] = "max_drop present in per_seed metrics."
    except Exception as e:
        out["pass"] = False
        out["message"] = str(e)
    return out


def check_selection_pool(plot2_dir: Path) -> Dict[str, Any]:
    """4) No duplicates in trained graph IDs; report method breakdown. Stage 0: no duplicate model_name; no duplicate graph_hash within each method (overlap across methods is allowed per spec)."""
    sel_path = plot2_dir / "selected_architectures.csv"
    out = {"path": str(sel_path), "exists": sel_path.exists(), "pass": False, "n_rows": 0, "n_unique_model_names": 0, "duplicates": [], "by_method": {}, "graph_hash_duplicates": []}
    if not sel_path.exists():
        out["message"] = "selected_architectures.csv not found."
        return out
    try:
        import pandas as pd
        df = pd.read_csv(sel_path)
        if "model_name" not in df.columns:
            out["message"] = "No model_name column."
            return out
        names = df["model_name"].astype(str).tolist()
        out["n_rows"] = len(names)
        out["n_unique_model_names"] = len(set(names))
        dupes = [n for n in set(names) if names.count(n) > 1]
        out["duplicates"] = dupes
        graph_hash_dupes = []
        if "graph_hash" in df.columns and "method" in df.columns:
            # Per-method: no duplicate graph_hash within same method (overlap across methods allowed per Plot2 spec)
            for method in df["method"].dropna().astype(str).unique():
                sub = df[df["method"].astype(str) == method]
                hashes = sub["graph_hash"].dropna().astype(str).tolist()
                within_dupes = [h for h in set(hashes) if hashes.count(h) > 1]
                graph_hash_dupes.extend(within_dupes)
            out["graph_hash_duplicates"] = list(set(graph_hash_dupes))
        out["pass"] = len(dupes) == 0 and len(out["graph_hash_duplicates"]) == 0
        if "method" in df.columns:
            out["by_method"] = df.groupby("method").size().to_dict()
        out["message"] = "No duplicate model_name or graph_hash within method." if out["pass"] else f"Duplicates: {dupes}; graph_hash_dupes: {out['graph_hash_duplicates']}"
    except Exception as e:
        out["message"] = str(e)
    return out


def check_stage1_overlap_and_regime(plot2_dir: Path) -> Dict[str, Any]:
    """Stage 1: Overlap A-B <= 75% or >=2 graphs differ; overlap A-C and B-C <= 50%; span >=2 regimes and >=3 (C,L) cells."""
    sel_path = plot2_dir / "selected_architectures.csv"
    manifest_path = plot2_dir / "plot2_manifest.json"
    out = {"pass": True, "overlap_A_B": float("nan"), "overlap_A_C": float("nan"), "overlap_B_C": float("nan"), "n_regimes": 0, "n_cells": 0, "message": ""}
    if not sel_path.exists() or not manifest_path.exists():
        out["pass"] = True
        out["message"] = "Missing selected_architectures or manifest; skip Stage 1."
        return out
    try:
        import pandas as pd
        df = pd.read_csv(sel_path)
        if "graph_hash" not in df.columns or "method" not in df.columns:
            out["message"] = "No graph_hash or method column; skip Stage 1."
            return out
        ws = df[df["wiring_kind"].astype(str) == "ws_flex"].copy() if "wiring_kind" in df.columns else df.copy()
        if ws.empty:
            out["message"] = "No ws_flex architectures; skip Stage 1."
            return out

        def _graphs(m: str) -> Set[str]:
            sub = ws[ws["method"].astype(str) == m]
            return set(sub["graph_hash"].dropna().astype(str).tolist())

        a_set = _graphs("baseline_a")
        b_set = _graphs("baseline_b")
        c_set = _graphs("tpe")
        if a_set and b_set:
            overlap_ab = len(a_set & b_set) / max(len(a_set), len(b_set))
            out["overlap_A_B"] = float(overlap_ab)
            if overlap_ab > 0.75 and len(a_set - b_set) + len(b_set - a_set) < 2:
                out["pass"] = False
                out["message"] = f"Stage 1: overlap(A,B)={overlap_ab:.2f} > 75% and <2 graphs differ."
        if a_set and c_set:
            overlap_ac = len(a_set & c_set) / max(len(a_set), len(c_set))
            out["overlap_A_C"] = float(overlap_ac)
            if overlap_ac > 0.50:
                out["pass"] = False
                out["message"] = out["message"] or f"Stage 1: overlap(A,C)={overlap_ac:.2f} > 50%."
        if b_set and c_set:
            overlap_bc = len(b_set & c_set) / max(len(b_set), len(c_set))
            out["overlap_B_C"] = float(overlap_bc)
            if overlap_bc > 0.50:
                out["pass"] = False
                out["message"] = out["message"] or f"Stage 1: overlap(B,C)={overlap_bc:.2f} > 50%."
        if "k" in ws.columns:
            regimes = set()
            deg = json.loads(manifest_path.read_text()).get("degree_regimes", {})
            for _, r in ws.iterrows():
                k = r.get("k")
                if pd.isna(k):
                    continue
                for name, ks in deg.items():
                    if int(k) in [int(x) for x in ks]:
                        regimes.add(str(name))
                        break
            out["n_regimes"] = len(regimes)
            if len(regimes) < 2:
                out["pass"] = False
                out["message"] = out["message"] or f"Stage 1: only {len(regimes)} regime(s); need >=2."
        if "C_bin" in ws.columns and "L_bin" in ws.columns:
            cells = set((str(r["C_bin"]), str(r["L_bin"])) for _, r in ws.iterrows() if pd.notna(r.get("C_bin")) and pd.notna(r.get("L_bin")))
            out["n_cells"] = len(cells)
            if len(cells) < 3:
                out["pass"] = False
                out["message"] = out["message"] or f"Stage 1: only {len(cells)} (C,L) cells; need >=3."
        if out["pass"]:
            out["message"] = "Stage 1: overlap and regime span OK."
    except Exception as e:
        out["pass"] = False
        out["message"] = str(e)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2 Phase 6: pipeline integrity checks")
    parser.add_argument("--plot2_dir", type=str, required=True, help="Plot 2 run directory")
    parser.add_argument("--out_json", type=str, default=None, help="Write report to this JSON file")
    parser.add_argument(
        "--allow_legacy",
        action="store_true",
        default=False,
        help="If set, missing perturbation fingerprint does not fail overall for runs predating PATCH 0.2. Default: False (all checks required).",
    )
    parser.add_argument("--strict", action="store_true", help="Require all checks including fingerprint (overrides --allow_legacy).")
    parser.add_argument(
        "--strict_spec",
        action="store_true",
        help="Spec §7: require fingerprint and for AR(1) runs lag1_autocorrelation >= 0.90. Implies strict, no legacy bypass.",
    )
    args = parser.parse_args()

    plot2_dir = Path(args.plot2_dir).resolve()
    if not plot2_dir.is_dir():
        print(f"Not a directory: {plot2_dir}")
        sys.exit(1)

    strict_spec = getattr(args, "strict_spec", False)
    if strict_spec:
        strict = True
        allow_legacy = False
    else:
        strict = getattr(args, "strict", False)
        allow_legacy = getattr(args, "allow_legacy", False) and not strict
    report = {
        "schema_version": 1,
        "plot2_dir": str(plot2_dir),
        "checks": {
            "perturbation_fingerprint": check_fingerprint(plot2_dir, strict_spec=strict_spec),
            "manifest_primary_perturbation": check_manifest_primary_perturbation(plot2_dir),
            "selection_pool_no_duplicates": check_selection_pool(plot2_dir),
            "stage1_overlap_and_regime": check_stage1_overlap_and_regime(plot2_dir),
            "collapse_scores": check_collapse_scores(plot2_dir),
            "max_drop_in_outputs": check_max_drop_in_outputs(plot2_dir),
        },
    }
    fp_pass = report["checks"]["perturbation_fingerprint"].get("pass", False)
    other_checks = {k: v for k, v in report["checks"].items() if k != "perturbation_fingerprint"}
    other_pass = all(c.get("pass", False) for c in other_checks.values())
    if allow_legacy and not fp_pass and other_pass:
        report["all_pass"] = True
        report["legacy_fingerprint_skipped"] = True
    else:
        report["all_pass"] = all(c.get("pass", False) for c in report["checks"].values())
        report["legacy_fingerprint_skipped"] = False

    for name, res in report["checks"].items():
        print(f"  {name}: {'PASS' if res.get('pass') else 'FAIL'} - {res.get('message', '')}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.out_json}")

    if report.get("legacy_fingerprint_skipped"):
        print("Overall: PASS (legacy run; fingerprint not required)")
    else:
        print(f"Overall: {'PASS' if report['all_pass'] else 'FAIL'}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
