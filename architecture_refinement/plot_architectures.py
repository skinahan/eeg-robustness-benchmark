#!/usr/bin/env python3
"""
Generate high-quality figures for:
  1) Multi-objective Pareto frontier from saved optimization results
  2) Wiring diagrams for top selected architectures (WS-flex hidden graph expanded to I/H/O)

Saves 300 dpi, labeled PNGs to the specified output directory.

Usage examples:
  - Pareto front from results JSON:
      python plot_architectures.py --results-json outputs/optimization/extended_demo_optimization_results.json --outdir outputs/plots

  - Wiring diagrams from exported best_graphs (GraphML) with specified I/O sizes:
      python plot_architectures.py --best-graphs-dir outputs/best_graphs --input-size 64 --output-size 2 --outdir outputs/plots --n-top 3

  - Wiring diagrams from saved architecture JSONs:
      python plot_architectures.py --architectures-dir outputs/architectures --outdir outputs/plots --n-top 3
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # When executed as a module: python -m architecture_refinement.plot_architectures
    from .arbitrary_wiring import WsFlexHiddenWiring
except Exception:
    # When executed as a script: python architecture_refinement/plot_architectures.py
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from architecture_refinement.arbitrary_wiring import WsFlexHiddenWiring


def set_style(figsize: Tuple[float, float] = (6.5, 4.5)) -> None:
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def plot_pareto_front(results: Dict[str, Any], outpath: str, x_key: str = "entropy", y_key: str = "curvature") -> None:
    """Plot background points (all) and highlight Pareto front solutions."""
    set_style((8, 4.0))

    all_pts = results.get("all_results", [])
    pareto = results.get("pareto_solutions", [])

    x_all = [r.get("objectives", {}).get(x_key, np.nan) for r in all_pts]
    y_all = [r.get("objectives", {}).get(y_key, np.nan) for r in all_pts]
    rs_all = [r.get("robustness_score", np.nan) for r in all_pts]

    plt.figure()
    sc = plt.scatter(x_all, y_all, cmap="viridis", alpha=0.5, label="All trials")
    # cb = plt.colorbar(sc)
    # cb.set_label("Robustness score")

    if pareto:
        x_pf = [p.get("objectives", {}).get(x_key, np.nan) for p in pareto]
        y_pf = [p.get("objectives", {}).get(y_key, np.nan) for p in pareto]
        plt.scatter(x_pf, y_pf, c="#d62728", edgecolor="k", s=70, label="Pareto front")

    plt.xlabel(x_key.title())
    plt.ylabel(y_key.title())
    plt.title(f"Pareto frontier: {x_key.title()} vs {y_key.title()}")
    # Move legend outside the plot area to avoid overlapping data
    # lgd = plt.legend(
    #     frameon=True,
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1.0),
    #     borderaxespad=0.0,
    # )
    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def _compute_layered_positions(I: int, H: int, O: int) -> Dict[int, Tuple[float, float]]:
    """Arrange nodes in three horizontal layers (inputs top, hidden middle, outputs bottom)."""
    def line_positions(n: int, y: float) -> List[Tuple[float, float]]:
        if n <= 0:
            return []
        xs = np.linspace(-1.0, 1.0, n)
        return [(float(x), float(y)) for x in xs]

    pos: Dict[int, Tuple[float, float]] = {}
    pos.update({i: p for i, p in enumerate(line_positions(I, 1.0))})
    pos.update({I + j: p for j, p in enumerate(line_positions(H, 0.0))})
    pos.update({I + H + k: p for k, p in enumerate(line_positions(O, -1.0))})
    return pos


def _compute_positions_with_hidden_force(
    W: np.ndarray,
    I: int,
    H: int,
    O: int,
    seed: int = 17,
    pull_to_io: bool = True,
    io_pull_weight: float = 1.0,
) -> Dict[int, Tuple[float, float]]:
    """Arrange inputs/outputs flat; hidden via force-directed layout.

    - Inputs at y=1.0, evenly spaced on x in [-1, 1]
    - Outputs at y=-1.0, evenly spaced on x in [-1, 1]
    - Hidden nodes positioned with spring_layout using only hidden↔hidden edges,
      then scaled to x in [-1,1], y in [-0.35, 0.35].
    """
    # Start with flat I/O
    pos = _compute_layered_positions(I, 0, O)

    if H <= 0:
        return pos

    # Build layout graph including hidden and (optionally) I/O attraction edges
    hidden_indices = list(range(I, I + H))
    input_indices = list(range(0, I))
    output_indices = list(range(I + H, I + H + O))

    G_layout = nx.Graph()
    # Add nodes
    G_layout.add_nodes_from(hidden_indices)
    if pull_to_io:
        G_layout.add_nodes_from(input_indices)
        G_layout.add_nodes_from(output_indices)

    # Hidden↔hidden edges (undirected for spacing)
    hh_block = W[I:I + H, I:I + H]
    srcs, dsts = np.where(hh_block > 0)
    for s, d in zip(srcs.tolist(), dsts.tolist()):
        if s == d:
            continue
        u = I + int(s)
        v = I + int(d)
        if u == v:
            continue
        G_layout.add_edge(u, v, weight=1.0)

    # I/O attraction edges: input→hidden and hidden→output
    if pull_to_io:
        # Input→Hidden
        ih_block = W[0:I, I:I + H]
        srcs, dsts = np.where(ih_block > 0)
        for s, d in zip(srcs.tolist(), dsts.tolist()):
            u = int(s)  # input idx
            v = I + int(d)  # hidden idx
            G_layout.add_edge(u, v, weight=float(max(io_pull_weight, 0.0)))

        # Hidden→Output
        ho_block = W[I:I + H, I + H:I + H + O]
        srcs, dsts = np.where(ho_block > 0)
        for s, d in zip(srcs.tolist(), dsts.tolist()):
            u = I + int(s)  # hidden idx
            v = I + H + int(d)  # output idx
            G_layout.add_edge(u, v, weight=float(max(io_pull_weight, 0.0)))

    if G_layout.number_of_nodes() == 0:
        # Fallback: keep hidden flat in the middle if something is off
        pos.update({I + j: (float(x), 0.0) for j, (x, _) in enumerate(_compute_layered_positions(0, H, 0).items())})
        return pos

    # Initial positions: flat I/O (if present), hidden in middle
    pos_init = {}
    # Inputs and outputs fixed on lines
    for i, (x, y) in _compute_layered_positions(I, 0, O).items():
        # The helper returns dict indices from 0..I-1 and then I+0.., but we need to map only inputs and outputs
        if i < I:
            pos_init[i] = (x, y)  # inputs at y=1
        else:
            # This branch would correspond to outputs if it were using the full layered call; we re-add below
            pass
    # Outputs
    for k, (x, y) in _compute_layered_positions(0, 0, O).items():
        pos_init[I + H + k] = (x, y)  # outputs at y=-1
    # Hidden start centered
    for j in hidden_indices:
        pos_init[j] = (0.0, 0.0)

    # Fixed nodes: inputs and outputs (when included)
    fixed_nodes = input_indices + output_indices if pull_to_io else []
    raw_pos = nx.spring_layout(
        G_layout,
        seed=seed,
        k=None,
        pos=pos_init if pull_to_io else None,
        fixed=fixed_nodes if pull_to_io else None,
        weight="weight",
    )

    # Extract arrays for hidden nodes and normalize to [-1,1] on x and [-a,a] on y
    xs = np.array([raw_pos[n][0] for n in hidden_indices])
    ys = np.array([raw_pos[n][1] for n in hidden_indices])

    def _safe_minmax(arr: np.ndarray, default_span: float = 1.0) -> Tuple[float, float]:
        if arr.size == 0:
            return -default_span, default_span
        a_min = float(np.min(arr))
        a_max = float(np.max(arr))
        if np.isclose(a_min, a_max):
            # Prevent collapse to a point
            eps = 1e-3
            return a_min - eps, a_max + eps
        return a_min, a_max

    x_min, x_max = _safe_minmax(xs)
    y_min, y_max = _safe_minmax(ys)

    def _scale(v: float, a_min: float, a_max: float, new_min: float, new_max: float) -> float:
        return new_min + (0.0 if np.isclose(a_min, a_max) else (v - a_min) * (new_max - new_min) / (a_max - a_min))

    y_span = 0.5
    for idx, n in enumerate(hidden_indices):
        x = _scale(xs[idx], x_min, x_max, -1.0, 1.0)
        y = _scale(ys[idx], y_min, y_max, -y_span, y_span)
        pos[n] = (float(x), float(y))

    # Put flattened inputs and outputs back explicitly (if any)
    for i, (x, y) in _compute_layered_positions(I, 0, 0).items():
        pos[i] = (x, y)
    for k, p in _compute_layered_positions(0, 0, O).items():
        pos[I + H + k] = p

    return pos


def draw_wiring_matrix(
    W: np.ndarray,
    I: int,
    H: int,
    O: int,
    title: str,
    outpath: str,
    layout: str = "force_hidden",
    seed: int = 17,
    size_by_degree: bool = True,
    hidden_size_min: float = 100.0,
    hidden_size_max: float = 320.0,
    io_size: float = 120.0,
    pull_to_io: bool = True,
    io_pull_weight: float = 1.0,
) -> None:
    """Draw the full wiring as a directed graph with I/H/O layers and colored edges."""
    set_style((7.5, 6.0))

    T = I + H + O
    # If metadata is inconsistent with the matrix, reconcile H from W
    if W.shape[0] != T:
        H_recalc = int(W.shape[0]) - int(I) - int(O)
        if H_recalc > 0:
            H = H_recalc
            T = I + H + O
        else:
            raise AssertionError(f"W shape {W.shape} incompatible with I={I}, O={O}")

    G = nx.DiGraph()
    for n in range(T):
        G.add_node(n)

    # Build edges from matrix (positive weights only for clarity)
    srcs, dsts = np.where(W > 0)
    for s, d in zip(srcs.tolist(), dsts.tolist()):
        G.add_edge(int(s), int(d))

    if layout == "force_hidden":
        pos = _compute_positions_with_hidden_force(
            W, I, H, O,
            seed=seed,
            pull_to_io=pull_to_io,
            io_pull_weight=io_pull_weight,
        )
    else:
        pos = _compute_layered_positions(I, H, O)

    # Node colors by layer
    node_colors = []
    for n in range(T):
        if n < I:
            node_colors.append("#1f77b4")  # inputs: blue
        elif n < I + H:
            node_colors.append("#2ca02c")  # hidden: green
        else:
            node_colors.append("#ff7f0e")  # outputs: orange

    # Edge colors by type
    edge_colors = []
    for u, v in G.edges():
        if u < I and I <= v < I + H:
            edge_colors.append("#4c78a8")  # input->hidden
        elif I <= u < I + H and I <= v < I + H:
            edge_colors.append("#72b7b2")  # hidden->hidden
        elif I <= u < I + H and v >= I + H:
            edge_colors.append("#f58518")  # hidden->output
        else:
            edge_colors.append("#9e9e9e")  # other (should be rare)

    # Compute node sizes (degree-based for hidden nodes if enabled)
    if size_by_degree:
        # Hidden↔hidden directed degree (in+out) from adjacency block
        hh_block = (W[I:I + H, I:I + H] > 0).astype(np.float32)
        if H > 0:
            hidden_deg = (hh_block.sum(axis=0) + hh_block.sum(axis=1)).A1 if hasattr(hh_block, 'A1') else (hh_block.sum(axis=0) + hh_block.sum(axis=1))
            hidden_deg = np.asarray(hidden_deg, dtype=np.float32).flatten()
            d_min = float(np.min(hidden_deg)) if hidden_deg.size > 0 else 0.0
            d_max = float(np.max(hidden_deg)) if hidden_deg.size > 0 else 0.0
            # Avoid zero span
            if np.isclose(d_min, d_max):
                hidden_sizes = np.full(H, (hidden_size_min + hidden_size_max) * 0.5, dtype=np.float32)
            else:
                hidden_sizes = hidden_size_min + (hidden_deg - d_min) * (hidden_size_max - hidden_size_min) / (d_max - d_min)
        else:
            hidden_sizes = np.array([], dtype=np.float32)

        node_sizes: List[float] = []
        for n in range(T):
            if n < I:
                node_sizes.append(float(io_size))
            elif n < I + H:
                node_sizes.append(float(hidden_sizes[n - I]))
            else:
                node_sizes.append(float(io_size))
    else:
        node_sizes = [float(io_size)] * T

    plt.figure()
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.5, edgecolors="k")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=False, width=0.8, alpha=0.7)

    # Layer labels
    plt.text(0.0, 1.15, f"Inputs (I={I})", ha="center", va="center", fontsize=11)
    plt.text(-1.15, 0.15, f"Hidden (H={H})", ha="left", va="center", fontsize=11)
    plt.text(0.0, -1.15, f"Outputs (O={O})", ha="center", va="center", fontsize=11)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def build_wiring_from_graphml(graphml_path: str, input_size: int, output_size: int, seed: int = 17) -> Tuple[np.ndarray, int, int, int]:
    G = nx.read_graphml(graphml_path)
    # Relabel to integer nodes if needed
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    wiring = WsFlexHiddenWiring(
        input_size=input_size,
        hidden_graph=G,
        output_size=output_size,
        input_strategy="degree_proportional",
        output_strategy="uniform",
        hidden_edge_orientation="random_oriented",
        add_hidden_self_loops=True,
        allow_signed_hidden_edges=True,
        inhibitory_ratio=0.2,
        seed=seed,
    )
    W = wiring.full_wiring_matrix()
    I, H, O = input_size, len(G.nodes()), output_size
    return W, I, H, O


def build_wiring_from_arch_json(arch_json_path: str) -> Tuple[np.ndarray, int, int, int]:
    with open(arch_json_path, "r") as f:
        data = json.load(f)
    I = int(data["input_size"])  # noqa
    H = int(data["hidden_size"])  # noqa
    O = int(data["output_size"])  # noqa
    A = np.asarray(data["wiring_matrix"], dtype=np.float32)
    T_expected = I + H + O

    # Case 1: JSON already contains full wiring matrix (T x T)
    if A.shape == (T_expected, T_expected):
        return A, I, H, O

    # Case 2: JSON contains hidden-only adjacency (H x H) -> build full wiring
    if A.shape == (H, H):
        wiring = WsFlexHiddenWiring(
            input_size=I,
            hidden_graph=A,
            output_size=O,
            input_strategy="degree_proportional",
            output_strategy="uniform",
            hidden_edge_orientation="random_oriented",
            add_hidden_self_loops=True,
            allow_signed_hidden_edges=True,
            inhibitory_ratio=0.2,
            seed=17,
        )
        W = wiring.full_wiring_matrix()
        return W, I, H, O

    # Otherwise, shapes are inconsistent with metadata
    raise ValueError(
        f"Unsupported wiring_matrix shape {A.shape}; expected (H,H)=({H},{H}) or (T,T)=({T_expected},{T_expected})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pareto and wiring figures (300 dpi)")
    parser.add_argument("--results-json", type=str, default=None, help="Path to optimizer results JSON (with pareto_solutions and all_results)")
    parser.add_argument("--best-graphs-dir", type=str, default=None, help="Directory with best_graph_*.graphml files")
    parser.add_argument("--architectures-dir", type=str, default=None, help="Directory with architecture JSONs")
    parser.add_argument("--input-size", type=int, default=None, help="Input size (if using GraphML best graphs)")
    parser.add_argument("--output-size", type=int, default=None, help="Output size (if using GraphML best graphs)")
    parser.add_argument("--n-top", type=int, default=3, help="How many architectures to visualize")
    parser.add_argument(
        "--layout",
        type=str,
        choices=["force_hidden", "layered"],
        default="force_hidden",
        help="Wiring layout: force_hidden keeps I/O flat and distributes hidden by connectivity",
    )
    parser.add_argument("--outdir", type=str, default="outputs/plots", help="Directory to save figures")
    parser.add_argument("--no-size-by-degree", action="store_true", help="Disable degree-based node sizing for hidden nodes")
    parser.add_argument("--hidden-size-min", type=float, default=100.0, help="Minimum hidden node size when sizing by degree")
    parser.add_argument("--hidden-size-max", type=float, default=320.0, help="Maximum hidden node size when sizing by degree")
    parser.add_argument("--io-size", type=float, default=120.0, help="Node size for input/output layers")
    parser.add_argument("--no-pull-to-io", action="store_true", help="Disable attraction of hidden nodes toward connected I/O nodes in force layout")
    parser.add_argument("--io-pull-weight", type=float, default=1.0, help="Relative strength of I/O attraction in force layout")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Pareto frontier
    if args.results_json and os.path.exists(args.results_json):
        results = load_results_json(args.results_json)
        plot_pareto_front(results, str(outdir / "pareto_frontier.png"))

    # 2) Wiring diagrams
    count = 0
    # From architecture JSONs (preferred: they carry I/H/O)
    if args.architectures_dir and os.path.isdir(args.architectures_dir):
        json_files = sorted([p for p in Path(args.architectures_dir).glob("*.json")])
        for idx, jf in enumerate(json_files[: args.n_top]):
            W, I, H, O = build_wiring_from_arch_json(str(jf))
            title = f"Architecture {idx+1}: I={I}, H={H}, O={O}"
            outpath = str(outdir / f"wiring_arch_{idx+1}.png")
            draw_wiring_matrix(
                W, I, H, O, title, outpath,
                layout=args.layout,
                size_by_degree=not args.no_size_by_degree,
                hidden_size_min=args.hidden_size_min,
                hidden_size_max=args.hidden_size_max,
                io_size=args.io_size,
                pull_to_io=not args.no_pull_to_io,
                io_pull_weight=args.io_pull_weight,
            )
            count += 1

    # From best_graphs GraphML (requires I/O sizes)
    if count < args.n_top and args.best_graphs_dir and os.path.isdir(args.best_graphs_dir):
        if args.input_size is None or args.output_size is None:
            raise SystemExit("--input-size and --output-size are required when using --best-graphs-dir")
        gml_files = sorted([p for p in Path(args.best_graphs_dir).glob("*.graphml")])
        remaining = args.n_top - count
        for idx, gf in enumerate(gml_files[: remaining]):
            W, I, H, O = build_wiring_from_graphml(str(gf), args.input_size, args.output_size)
            title = f"Best graph {idx+1}: I={I}, H={H}, O={O}"
            outpath = str(outdir / f"wiring_best_graph_{idx+1}.png")
            draw_wiring_matrix(
                W, I, H, O, title, outpath,
                layout=args.layout,
                size_by_degree=not args.no_size_by_degree,
                hidden_size_min=args.hidden_size_min,
                hidden_size_max=args.hidden_size_max,
                io_size=args.io_size,
                pull_to_io=not args.no_pull_to_io,
                io_pull_weight=args.io_pull_weight,
            )


if __name__ == "__main__":
    main()


