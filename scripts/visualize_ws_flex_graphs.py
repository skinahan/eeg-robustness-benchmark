#!/usr/bin/env python3
"""
Visualize WS-Flex graphs from the architecture refinement process.

Uses the same layout rules as plot_architectures.py: spring_layout so that
connected nodes appear near each other and unconnected nodes are pushed apart.
Saves 300 dpi figures (PNG and PDF).

Usage:
  - From GraphML files:
      python scripts/visualize_ws_flex_graphs.py --graphs outputs/first_pass_summary/best_graph_1_trial_365.graphml outputs/best_graphs/best_graph_2_trial_219.graphml --outdir outputs/ws_flex_figures

  - From architecture refinement outputs:
      python scripts/visualize_ws_flex_graphs.py --best-graphs-dir outputs/first_pass_summary --outdir outputs/ws_flex_figures --n-top 4

  - Generate from params (H, k, p, seed):
      python scripts/visualize_ws_flex_graphs.py --generate --H 32 --k 6 --p 0.2 --seed 42 --outdir outputs/ws_flex_figures

  - Full wiring with I/O layers (input=blue, hidden=green, output=orange):
      python scripts/visualize_ws_flex_graphs.py --best-graphs-dir outputs/first_pass_summary --input-size 64 --output-size 2 --outdir outputs/ws_flex_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Add project root
import sys
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def _compute_ws_flex_layout(
    G: nx.Graph,
    seed: int = 17,
    k: Optional[float] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Position nodes using spring_layout (same rules as plot_architectures.py).

    Connected nodes are pulled together; unconnected nodes are pushed apart.
    Positions are scaled to [-1, 1] on both axes for consistent figure size.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    # Use spring_layout: edges act as springs (connected = near), nodes repel (unconnected = far)
    # Same rule as plot_architectures/demo: k controls spacing (larger = nodes farther apart)
    if k is None:
        k = 1.5 if n <= 32 else (1.0 if n <= 64 else 1.5 / np.sqrt(n))
    pos = nx.spring_layout(G, seed=seed, k=float(k), iterations=100)

    # Convert to integer keys if needed (GraphML may use string ids)
    nodes = list(G.nodes())
    xs = np.array([pos[n][0] for n in nodes])
    ys = np.array([pos[n][1] for n in nodes])

    def _safe_minmax(arr: np.ndarray, default_span: float = 1.0) -> Tuple[float, float]:
        if arr.size == 0:
            return -default_span, default_span
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if np.isclose(a_min, a_max):
            return a_min - 1e-3, a_max + 1e-3
        return a_min, a_max

    x_min, x_max = _safe_minmax(xs)
    y_min, y_max = _safe_minmax(ys)

    def _scale(v: float, lo: float, hi: float, new_lo: float, new_hi: float) -> float:
        if np.isclose(lo, hi):
            return (new_lo + new_hi) / 2.0
        return new_lo + (v - lo) * (new_hi - new_lo) / (hi - lo)

    result: Dict[int, Tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        x = _scale(xs[i], x_min, x_max, -1.0, 1.0)
        y = _scale(ys[i], y_min, y_max, -1.0, 1.0)
        # Use integer index for drawing (nx expects node id as key)
        result[node] = (float(x), float(y))

    return result


def load_graph_from_graphml(path: str) -> nx.Graph:
    """Load graph from GraphML and ensure integer node labels."""
    G = nx.read_graphml(path)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def draw_ws_flex_graph(
    G: nx.Graph,
    outpath: Path,
    title: Optional[str] = None,
    seed: int = 17,
    node_size: float = 120.0,
    node_color: str = "#2ca02c",
    edge_color: str = "#72b7b2",
    edge_alpha: float = 0.6,
    dpi: int = 300,
) -> None:
    """Draw a single WS-Flex graph (hidden-only) and save."""
    outpath.parent.mkdir(parents=True, exist_ok=True)

    pos = _compute_ws_flex_layout(G, seed=seed)

    # Optional: size nodes by degree
    degrees = dict(G.degree())
    if degrees:
        d_min = min(degrees.values())
        d_max = max(degrees.values())
        if d_max > d_min:
            sizes = [80 + 120 * (degrees[n] - d_min) / (d_max - d_min) for n in G.nodes()]
        else:
            sizes = [node_size] * G.number_of_nodes()
    else:
        sizes = [node_size] * G.number_of_nodes()

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color,
        node_size=sizes,
        linewidths=0.5,
        edgecolors="k",
        ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_color,
        alpha=edge_alpha,
        width=0.8,
        ax=ax,
    )

    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        p = outpath.with_suffix(f".{ext}")
        plt.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close()


def draw_full_wiring(
    W: np.ndarray,
    I: int,
    H: int,
    O: int,
    outpath: Path,
    title: Optional[str] = None,
    seed: int = 17,
    dpi: int = 300,
) -> None:
    """Draw full I/H/O wiring with input (blue), hidden (green), output (orange)."""
    from architecture_refinement.plot_architectures import draw_wiring_matrix
    outpath.parent.mkdir(parents=True, exist_ok=True)
    t = title or f"WS-Flex wiring: I={I}, H={H}, O={O}"
    for ext in ["png", "pdf"]:
        p = outpath.with_suffix(f".{ext}")
        draw_wiring_matrix(
            W, I, H, O,
            title=t,
            outpath=str(p),
            layout="force_hidden",
            seed=seed,
            size_by_degree=True,
            pull_to_io=True,
        )
        print(f"Saved: {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize WS-Flex graphs (300 dpi)")
    parser.add_argument("--graphs", nargs="+", type=str, help="Paths to GraphML files")
    parser.add_argument("--best-graphs-dir", type=str, help="Directory with best_graph_*.graphml files")
    parser.add_argument("--n-top", type=int, default=4, help="Number of graphs when using --best-graphs-dir")
    parser.add_argument("--generate", action="store_true", help="Generate graph from params instead of loading")
    parser.add_argument("--input-size", type=int, default=None, help="Input size; with --output-size, draw full I/H/O with colored layers")
    parser.add_argument("--output-size", type=int, default=None, help="Output size; with --input-size, draw full I/H/O with colored layers")
    parser.add_argument("--H", type=int, default=32, help="Hidden size (for --generate)")
    parser.add_argument("--k", type=int, default=6, help="Neighbors per node (for --generate)")
    parser.add_argument("--p", type=float, default=0.2, help="Rewiring probability (for --generate)")
    parser.add_argument("--seed", type=int, default=42, help="Graph seed (for --generate or layout)")
    parser.add_argument("--outdir", type=str, default="outputs/ws_flex_figures", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    graph_paths: List[Path] = []

    if args.generate:
        from architecture_refinement.ws_flex_generator import build_plain_ws_flex
        from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring
        G = build_plain_ws_flex(args.H, args.k, args.p, args.seed)
        outpath = outdir / f"ws_flex_H{args.H}_k{args.k}_p{args.p}_seed{args.seed}"
        if args.input_size is not None and args.output_size is not None:
            wiring = WsFlexHiddenWiring(
                input_size=args.input_size,
                hidden_graph=G,
                output_size=args.output_size,
                input_strategy="degree_proportional",
                output_strategy="uniform",
                hidden_edge_orientation="random_oriented",
                add_hidden_self_loops=True,
                allow_signed_hidden_edges=True,
                inhibitory_ratio=0.2,
                seed=args.seed,
            )
            W = wiring.full_wiring_matrix()
            draw_full_wiring(
                W, args.input_size, args.H, args.output_size,
                outpath,
                title=f"WS-Flex: H={args.H}, k={args.k}, p={args.p}",
                seed=17,
            )
        else:
            draw_ws_flex_graph(
                G,
                outpath,
                title=f"WS-Flex: H={args.H}, k={args.k}, p={args.p}",
                seed=17,
            )
        return

    if args.graphs:
        graph_paths = [Path(p) for p in args.graphs if Path(p).exists()]
    elif args.best_graphs_dir:
        d = Path(args.best_graphs_dir)
        if d.is_dir():
            graph_paths = sorted(d.glob("best_graph_*.graphml"))[: args.n_top]

    if not graph_paths:
        print("No GraphML files found. Use --graphs, --best-graphs-dir, or --generate.")
        return

    use_full_wiring = args.input_size is not None and args.output_size is not None
    if use_full_wiring:
        from architecture_refinement.plot_architectures import build_wiring_from_graphml
        for i, gp in enumerate(graph_paths):
            W, I, H, O = build_wiring_from_graphml(str(gp), args.input_size, args.output_size, seed=args.seed)
            title = f"WS-Flex: {gp.stem} (I={I}, H={H}, O={O})"
            outpath = outdir / f"ws_flex_wiring_{i + 1}_{gp.stem}"
            draw_full_wiring(W, I, H, O, outpath, title=title, seed=args.seed)
    else:
        for i, gp in enumerate(graph_paths):
            G = load_graph_from_graphml(str(gp))
            n = G.number_of_nodes()
            m = G.number_of_edges()
            title = f"WS-Flex: {gp.stem} (n={n}, m={m})"
            outpath = outdir / f"ws_flex_{i + 1}_{gp.stem}"
            draw_ws_flex_graph(G, outpath, title=title, seed=args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
