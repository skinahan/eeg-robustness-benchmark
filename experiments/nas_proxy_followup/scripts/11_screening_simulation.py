#!/usr/bin/env python3
"""Top-k screening vs random baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.analysis.screening import simulate_topk_screening
from src.config_util import add_config_args, load_merged_configs, parse_config_paths, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_merged_configs(parse_config_paths(args))

    import pandas as pd

    merged_path = resolve_path(
        cfg,
        "merged_table",
        "experiments/nas_proxy_followup/outputs/analysis/merged_run_table.csv",
    )
    df = pd.read_csv(merged_path)
    out_dir = resolve_path(cfg, "analysis_output_dir", "experiments/nas_proxy_followup/outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    proxy_col = cfg.get("screening_proxy_column", "s_jac_mean_mean")
    target_col = cfg.get("screening_target_column", "test_auc")
    if proxy_col not in df.columns:
        proxy_col = df.columns[0]
    if target_col not in df.columns:
        target_col = df.columns[-1]

    q_values = [float(x) for x in cfg.get("screening_q_values", [0.05, 0.1, 0.2])]
    sim = simulate_topk_screening(df, proxy_col, target_col, q_values=q_values, num_random_trials=int(cfg.get("screening_random_trials", 500)))
    sim.to_csv(out_dir / "screening_summary.csv", index=False)
    print(f"Wrote {out_dir / 'screening_summary.csv'}")


if __name__ == "__main__":
    main()
