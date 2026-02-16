from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import math


Point2D = Tuple[float, float]


def pareto_front_2d(points: Iterable[Point2D]) -> List[Point2D]:
    """
    Compute the Pareto front for a 2D maximize-maximize problem.

    Returns a list of non-dominated points sorted by descending x.

    For 2D max-max, we can sort by x desc then keep points with strictly
    increasing y (this removes dominated points).
    """
    pts: List[Point2D] = []
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        pts.append((float(x), float(y)))

    if not pts:
        return []

    # Sort by x desc, then y desc (so ties keep best y first).
    pts.sort(key=lambda p: (-p[0], -p[1]))

    front: List[Point2D] = []
    best_y = -math.inf
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return front


def hypervolume_2d(points_pf: Sequence[Point2D], ref: Point2D = (-0.05, -0.05)) -> float:
    """
    Compute 2D hypervolume for a maximize-maximize Pareto front with a reference point.

    Assumes `points_pf` are (approximately) non-dominated. Works even if not perfectly.
    """
    rx, ry = float(ref[0]), float(ref[1])
    pts = [(float(x), float(y)) for x, y in points_pf if math.isfinite(x) and math.isfinite(y)]
    if not pts:
        return 0.0

    # Ensure sorted by x descending; y should be non-decreasing in this order for a PF.
    pts.sort(key=lambda p: -p[0])

    hv = 0.0
    # Append a terminal x at the reference point for the final strip.
    xs = [p[0] for p in pts] + [rx]
    ys = [p[1] for p in pts]

    # Clamp y to be at least ref_y for safety
    best_y = ry
    for i in range(len(ys)):
        x_i = xs[i]
        x_next = xs[i + 1]
        y_i = ys[i]
        if y_i > best_y:
            best_y = y_i
        width = max(0.0, x_i - x_next)
        height = max(0.0, best_y - ry)
        hv += width * height

    return float(hv)

