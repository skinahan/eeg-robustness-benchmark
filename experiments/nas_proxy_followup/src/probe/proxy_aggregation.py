"""Aggregate probe metrics across seeds."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


def aggregate_probe_seed_metrics(probe_metric_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not probe_metric_rows:
        return {}
    keys = [k for k in probe_metric_rows[0] if k not in ("probe_seed", "topology_id", "mapping_scheme")]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [float(r[k]) for r in probe_metric_rows if k in r and r[k] is not None]
        if vals:
            out[f"{k}_mean"] = mean(vals)
            if len(vals) > 1:
                out[f"{k}_std"] = pstdev(vals)
    return out
