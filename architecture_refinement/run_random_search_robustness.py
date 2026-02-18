#!/usr/bin/env python3
"""
Random search over the same WS-flex parameter space as the first pass.
Samples N graphs, computes robustness scores via TE + ORC proxy metrics,
and outputs the robustness distribution with *_params.json files.

Usage:
    python -m architecture_refinement.run_random_search_robustness \
        --n_samples 365 \
        --output_dir outputs/random_search_robustness \
        --seed 42
"""
# Avoid OMP duplicate library crash on Windows (Intel MKL / OpenMP)
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import networkx as nx

# Add parent for imports when run as script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from architecture_refinement.config import Config
from architecture_refinement.graph_generator import ModularSmallWorldGraphGenerator
from architecture_refinement.topology_analyzer import TopologyAnalyzer


def _compute_robustness_score(metrics: dict, topology_analyzer: TopologyAnalyzer) -> float:
    """Compute robustness score from metrics (TE + ORC). Same formula as topology_analyzer."""
    if hasattr(topology_analyzer, "compute_robustness_score"):
        return topology_analyzer.compute_robustness_score(metrics)
    # Fallback: replicate topology_analyzer.compute_robustness_score logic
    te = float(np.clip(metrics.get("te", 0.0), 0.0, 1.0))
    orc_val = float(metrics.get("orc", metrics.get("avg_ricci_curvature", 0.0)))
    curvature_score = float(1.0 / (1.0 + np.exp(-orc_val)))
    return 0.5 * te + 0.5 * curvature_score


def _sample_parameters(
    i: int,
    rng: np.random.Generator,
    graph_config: Any,
    opt_config: Any,
) -> Dict[str, Any]:
    """Sample one parameter set from the same bounds as optimizer._suggest_parameters."""
    min_units = graph_config.min_units
    max_units = graph_config.max_units
    min_k = opt_config.min_k_degree
    max_k = opt_config.max_k_degree
    min_p = opt_config.min_p_rewiring
    max_p = opt_config.max_p_rewiring

    units = int(rng.integers(min_units, max_units + 1))
    out_lo = max(2, units // 10)
    out_hi = min(units // 3, 20)
    if out_lo > out_hi:
        out_lo, out_hi = out_hi, out_lo
    output_size = int(rng.integers(out_lo, out_hi + 1)) if out_lo <= out_hi else out_lo

    target_clustering = float(rng.uniform(0.01, 1.0))
    target_path_length = float(rng.uniform(1.0, 4.5))
    k_degree = int(rng.integers(min_k, max_k + 1))
    k_degree = max(2, min(k_degree, units // 2))  # WS constraint
    p_rewiring = float(rng.uniform(min_p, max_p))

    return {
        "units": units,
        "output_size": output_size,
        "target_clustering": target_clustering,
        "target_path_length": target_path_length,
        "k_degree": k_degree,
        "p_rewiring": p_rewiring,
        "seed": i,
    }


def run_random_search(
    n_samples: int = 365,
    output_dir: Path = None,
    seed: int = 42,
    n_top: int = 5,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """
    Run random search over WS-flex parameter space and compute robustness distribution.

    Returns:
        Summary dict with robustness_stats, n_samples, and paths.
    """
    logger = logger or logging.getLogger(__name__)
    output_dir = output_dir or Path("outputs/random_search_robustness")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    graph_generator = ModularSmallWorldGraphGenerator(config.graph_generation, logger=logger)
    topology_analyzer = TopologyAnalyzer(config.topology_metrics, logger=logger)

    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []

    logger.info(f"Running random search: n_samples={n_samples}, seed={seed}")
    logger.info(f"Parameter bounds: units {config.graph_generation.min_units}-{config.graph_generation.max_units}, "
                f"k_degree {config.optimization.min_k_degree}-{config.optimization.max_k_degree}, "
                f"p_rewiring {config.optimization.min_p_rewiring}-{config.optimization.max_p_rewiring}")

    for i in range(n_samples):
        try:
            params = _sample_parameters(
                i, rng, config.graph_generation, config.optimization
            )
            graph = graph_generator._create_watts_strogatz_flex_graph(
                params["units"],
                params["k_degree"],
                params["p_rewiring"],
                params["target_clustering"],
                params["target_path_length"],
                seed=params["seed"],
            )
            if graph is None or graph.number_of_nodes() == 0:
                logger.warning(f"Sample {i}: empty graph, skipping")
                continue

            metrics = topology_analyzer.analyze_graph(graph)
            robustness_score = _compute_robustness_score(metrics, topology_analyzer)

            entropy = float(np.clip(metrics.get("te", 0.0), 0.0, 1.0))
            curvature = float(metrics.get("orc", metrics.get("avg_ricci_curvature", 0.0)))

            result = {
                "trial_number": i,
                "parameters": params,
                "objectives": {"entropy": entropy, "curvature": curvature},
                "robustness_score": float(robustness_score),
                "graph": graph,
            }
            results.append(result)

            # Save per-sample params (without graph)
            param_file = output_dir / f"random_sample_{i}_params.json"
            save_obj = {
                "trial_number": result["trial_number"],
                "parameters": result["parameters"],
                "objectives": result["objectives"],
                "robustness_score": result["robustness_score"],
            }
            with open(param_file, "w") as f:
                json.dump(save_obj, f, indent=2)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{n_samples} samples")

        except Exception as e:
            logger.warning(f"Sample {i} failed: {e}")
            continue

    if not results:
        logger.error("No valid results; cannot compute distribution")
        return {"n_samples": 0, "error": "No valid results"}

    # Sort by robustness_score descending (same as get_best_solutions)
    results_sorted = sorted(results, key=lambda r: r["robustness_score"], reverse=True)

    # Robustness distribution summary
    scores = [r["robustness_score"] for r in results]
    summary = {
        "n_samples": len(results),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "percentiles": {
            "25": float(np.percentile(scores, 25)),
            "50": float(np.percentile(scores, 50)),
            "75": float(np.percentile(scores, 75)),
            "90": float(np.percentile(scores, 90)),
            "95": float(np.percentile(scores, 95)),
        },
    }

    summary_path = output_dir / "robustness_distribution_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {summary_path}")

    # Export top N by robustness (same format as first pass)
    for rank, result in enumerate(results_sorted[:n_top], start=1):
        trial_num = result["trial_number"]
        save_obj = {
            "trial_number": trial_num,
            "parameters": result["parameters"],
            "objectives": result["objectives"],
            "robustness_score": result["robustness_score"],
        }
        param_file = output_dir / f"best_graph_{rank}_trial_{trial_num}_params.json"
        with open(param_file, "w") as f:
            json.dump(save_obj, f, indent=2)

        if "graph" in result and result["graph"] is not None:
            graph_file = output_dir / f"best_graph_{rank}_trial_{trial_num}.graphml"
            nx.write_graphml(result["graph"], graph_file)

    logger.info(f"Exported top {n_top} graphs by robustness_score")
    logger.info(f"Robustness distribution: mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
                f"min={summary['min']:.4f}, max={summary['max']:.4f}")

    # Optional: histogram plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(summary["mean"], color="red", linestyle="--", label=f"mean={summary['mean']:.3f}")
        ax.axvline(summary["median"], color="orange", linestyle=":", label=f"median={summary['median']:.3f}")
        ax.set_xlabel("Robustness score")
        ax.set_ylabel("Count")
        ax.set_title(f"Random Search Robustness Distribution (n={len(results)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        hist_path = output_dir / "robustness_distribution_histogram.png"
        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Wrote {hist_path}")
    except ImportError:
        logger.debug("matplotlib not available, skipping histogram")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Random search robustness distribution over WS-flex parameter space"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=365,
        help="Number of random samples (default: 365)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/random_search_robustness",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_top",
        type=int,
        default=5,
        help="Number of top graphs to export (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    summary = run_random_search(
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        n_top=args.n_top,
        logger=logger,
    )

    if summary.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
