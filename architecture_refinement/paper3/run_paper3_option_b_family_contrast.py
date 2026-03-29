"""
Paper 3 Option B: out-of-family topology contrast (G1–G5).

Aggregates experiment3_results.csv to one RD_max per (group, model), runs Kruskal–Wallis
(omnibus, topology-level units) and exploratory Mann–Whitney tests vs pooled WS-Flex (G1+G2).

Caveat: legacy Experiment 2 often has n_topologies=1 for G3–G5; pass ``--experiment2-dir``
if you used ``run_paper3_experiment2 --n-per-family N`` for matched counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architecture_refinement.paper3.run_paper3_q3_stratified_analysis import _bootstrap_group_mean_ci

GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5"]
GROUP_LABELS = {
    "G1": "G1 Proxy WS-Flex",
    "G2": "G2 Uniform WS-Flex",
    "G3": "G3 Dense CfC",
    "G4": "G4 Random sparse",
    "G5": "G5 NCP",
}


def topology_means_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (group, model): mean RD_max over seeds (matches selection comparison)."""
    sub = df.dropna(subset=["RD_max", "group", "model"])
    return sub.groupby(["group", "model"], as_index=False)["RD_max"].mean()


def run_option_b_analysis(df: pd.DataFrame, manifest_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    per_topo = topology_means_table(df)
    warnings: List[str] = []

    group_arrays: Dict[str, np.ndarray] = {}
    for g in GROUP_ORDER:
        v = per_topo[per_topo["group"] == g]["RD_max"].to_numpy(dtype=float)
        group_arrays[g] = v
        if len(v) < 2:
            warnings.append(
                f"{g} has n_topologies={len(v)}; omnibus and some pairwise tests have low or undefined power."
            )

    # Kruskal–Wallis: one sample per topology per group
    arrays_for_kw: List[np.ndarray] = []
    labels_kw: List[str] = []
    for g in GROUP_ORDER:
        arr = group_arrays[g]
        if len(arr) > 0:
            arrays_for_kw.append(arr)
            labels_kw.append(g)

    omnibus: Dict[str, Any] = {"test": "kruskal_wallis", "note": "Topology-level RD_max means per group."}
    if len(arrays_for_kw) >= 2:
        try:
            from scipy.stats import kruskal

            stat, pval = kruskal(*arrays_for_kw)
            omnibus["statistic"] = float(stat)
            omnibus["pvalue"] = float(pval)
            omnibus["groups_included"] = labels_kw
            omnibus["n_per_group"] = {labels_kw[i]: int(len(arrays_for_kw[i])) for i in range(len(labels_kw))}
        except ImportError:
            omnibus["error"] = "scipy not available"
    else:
        omnibus["error"] = "Need at least two nonempty groups for Kruskal–Wallis"

    pooled_ws = np.concatenate([group_arrays["G1"], group_arrays["G2"]])
    if len(pooled_ws) == 0:
        warnings.append("No G1/G2 data; cannot form pooled WS-Flex.")

    pairwise_vs_ws: Dict[str, Any] = {}
    for g in ["G3", "G4", "G5"]:
        other = group_arrays[g]
        if len(pooled_ws) < 1 or len(other) < 1:
            pairwise_vs_ws[g] = {"note": "missing data"}
            continue
        try:
            from scipy.stats import mannwhitneyu

            stat, pval = mannwhitneyu(pooled_ws, other, alternative="two-sided")
            pairwise_vs_ws[g] = {
                "comparison": f"pooled_G1_G2_vs_{g}",
                "exploratory": True,
                "mannwhitney_statistic": float(stat),
                "mannwhitney_pvalue": float(pval),
                "n_pooled_ws": int(len(pooled_ws)),
                "n_other": int(len(other)),
                "mean_pooled_ws": float(np.mean(pooled_ws)),
                "mean_other": float(np.mean(other)),
            }
        except (ImportError, ValueError) as e:
            pairwise_vs_ws[g] = {"error": str(e)}

    # Exploratory: among G3, G4, G5 only (single topologies common)
    baseline_pairwise: Dict[str, Any] = {}
    baselines = ["G3", "G4", "G5"]
    for i, a in enumerate(baselines):
        for b in baselines[i + 1 :]:
            va, vb = group_arrays[a], group_arrays[b]
            if len(va) < 1 or len(vb) < 1:
                continue
            key = f"{a}_vs_{b}"
            try:
                from scipy.stats import mannwhitneyu

                stat, pval = mannwhitneyu(va, vb, alternative="two-sided")
                baseline_pairwise[key] = {
                    "exploratory": True,
                    "mannwhitney_statistic": float(stat),
                    "mannwhitney_pvalue": float(pval),
                    "n_a": int(len(va)),
                    "n_b": int(len(vb)),
                }
            except (ImportError, ValueError) as e:
                baseline_pairwise[key] = {"error": str(e)}

    out: Dict[str, Any] = {
        "warnings": warnings,
        "caveat": (
            "G3/G4/G5 are often a single graph each; G1/G2 are many WS-Flex graphs. "
            "Between-family comparison mixes one realization vs a distribution for WS-Flex."
        ),
        "omnibus": omnibus,
        "pairwise_pooled_ws_vs_g3_g4_g5": pairwise_vs_ws,
        "pairwise_baselines_g3_g4_g5": baseline_pairwise,
    }
    if manifest_meta:
        out["experiment2_manifest_meta"] = manifest_meta
        if manifest_meta.get("matched_families_option_b") and manifest_meta.get("n_per_family"):
            out["caveat_matched"] = (
                f"Matched design: n_per_family={manifest_meta['n_per_family']} per G1–G5 "
                "(see experiment2_manifest.json). Omnibus tests use comparable topology counts."
            )
            out["warnings"] = [w for w in warnings if "n_topologies=1" not in w]
    return out


def plot_option_b_bars(
    df: pd.DataFrame,
    output_path: Path,
    *,
    matched_n_per_family: Optional[int] = None,
) -> None:
    """Bar chart aligned with analyze_selection_comparison + Option B title/annotation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_topo = topology_means_table(df)
    heights: List[float] = []
    errs_lo: List[float] = []
    errs_hi: List[float] = []
    labels: List[str] = []
    ns: List[int] = []

    for g in GROUP_ORDER:
        sub = per_topo[per_topo["group"] == g]
        vals = sub["RD_max"].to_numpy(dtype=float)
        n_topo = len(vals)
        if n_topo == 0:
            continue
        if g == "G5" and n_topo == 1:
            seed_vals = df[df["group"] == g]["RD_max"].to_numpy(dtype=float)
            m = float(np.mean(seed_vals))
            sem = float(np.std(seed_vals, ddof=1) / np.sqrt(len(seed_vals))) if len(seed_vals) > 1 else 0.0
            heights.append(m)
            errs_lo.append(m - 1.96 * sem)
            errs_hi.append(m + 1.96 * sem)
        else:
            m, lo, hi = _bootstrap_group_mean_ci(vals)
            heights.append(m)
            errs_lo.append(lo)
            errs_hi.append(hi)
        labels.append(GROUP_LABELS[g])
        ns.append(n_topo)

    if not heights:
        return

    x = np.arange(len(heights))
    yerr = np.array([[h - lo for h, lo in zip(heights, errs_lo)], [hi - h for h, hi in zip(heights, errs_hi)]])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    ax.bar(x, heights, yerr=yerr, capsize=4, color=colors[: len(heights)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean RD_max (topology mean, then group mean)")
    ax.set_title("Option B: out-of-family topology contrast (G1–G5)")
    if matched_n_per_family is not None:
        foot = (
            f"Matched design: n={matched_n_per_family} topologies per G1–G5 (Option B retrain). "
            "Compare groups at topology level."
        )
    else:
        foot = (
            "Note: G3–G5 are often n=1 topology vs many WS-Flex graphs; interpret as exploratory."
        )
    ax.text(0.5, -0.22, foot, transform=ax.transAxes, ha="center", fontsize=8, wrap=True)
    for i, n in enumerate(ns):
        ax.text(i, errs_hi[i] + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-6), f"n={n}", ha="center", fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Option B] Saved {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 Option B family contrast analysis")
    parser.add_argument(
        "--experiment3-csv",
        type=str,
        default="architecture_refinement/outputs/paper3/experiment3/experiment3_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_refinement/outputs/paper3/option_b",
    )
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--experiment2-dir",
        type=str,
        default=None,
        help="If set, read experiment2_manifest.json for matched-families metadata (option_b JSON).",
    )
    args = parser.parse_args()

    exp3 = Path(args.experiment3_csv)
    if not exp3.is_absolute():
        exp3 = _REPO_ROOT / exp3
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(exp3)
    per_topo = topology_means_table(df)
    per_topo.to_csv(out / "option_b_family_topology_means.csv", index=False)

    manifest_meta: Optional[Dict[str, Any]] = None
    if args.experiment2_dir:
        e2 = Path(args.experiment2_dir)
        if not e2.is_absolute():
            e2 = _REPO_ROOT / e2
        man_path = e2 / "experiment2_manifest.json"
        if man_path.is_file():
            full = json.loads(man_path.read_text(encoding="utf-8"))
            manifest_meta = {
                "matched_families_option_b": full.get("matched_families_option_b"),
                "n_per_family": full.get("n_per_family"),
                "K": full.get("K"),
                "S": full.get("S"),
            }

    result = run_option_b_analysis(df, manifest_meta=manifest_meta)
    result["inputs"] = {"experiment3_csv": str(exp3)}
    (out / "option_b_family_contrast.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[Option B] Wrote {out / 'option_b_family_topology_means.csv'}")
    print(f"[Option B] Wrote {out / 'option_b_family_contrast.json'}")

    if args.figures:
        mn = None
        if manifest_meta and manifest_meta.get("n_per_family"):
            mn = int(manifest_meta["n_per_family"])
        plot_option_b_bars(df, out / "plot_option_b_family_contrast.pdf", matched_n_per_family=mn)

    return 0


if __name__ == "__main__":
    sys.exit(main())
