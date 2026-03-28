#!/usr/bin/env python3
"""Proxy vs structural correlation and regression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_NPF_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (_REPO_ROOT, _NPF_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.analysis.proxy_validation import run_proxy_validation
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

    proxy_cols = [c for c in df.columns if "s_jac" in c or "s_fd" in c or c.endswith("_mean")]
    target_cols = [c for c in df.columns if "auc" in c.lower() or "degradation" in c.lower()]
    corr, reg = run_proxy_validation(df, proxy_cols[:20], target_cols[:20])
    corr.to_csv(out_dir / "proxy_correlations.csv", index=False)
    reg.to_csv(out_dir / "proxy_regression_summary.csv", index=False)
    print(f"Wrote {out_dir / 'proxy_correlations.csv'}")


if __name__ == "__main__":
    main()
