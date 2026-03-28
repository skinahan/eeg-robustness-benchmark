"""Correlations and optional regression between proxies and downstream targets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def run_proxy_validation(
    merged_df: pd.DataFrame,
    proxy_cols: List[str],
    target_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    for pc in proxy_cols:
        for tc in target_cols:
            if pc not in merged_df.columns or tc not in merged_df.columns:
                continue
            sub = merged_df[[pc, tc]].dropna()
            if len(sub) < 3:
                continue
            rho, p = stats.spearmanr(sub[pc], sub[tc])
            corr_rows.append(
                {
                    "proxy": pc,
                    "target": tc,
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n": len(sub),
                }
            )
    corr_df = pd.DataFrame(corr_rows)

    reg_rows = []
    try:
        import statsmodels.api as sm

        for tc in target_cols:
            if tc not in merged_df.columns:
                continue
            cols = [c for c in proxy_cols if c in merged_df.columns]
            if not cols:
                continue
            sub = merged_df[cols + [tc]].dropna()
            if len(sub) < len(cols) + 2:
                continue
            y = sub[tc].values
            X = sm.add_constant(sub[cols].values)
            res = sm.OLS(y, X).fit()
            reg_rows.append({"target": tc, "r2": float(res.rsquared), "n": len(sub)})
    except Exception:
        pass
    reg_df = pd.DataFrame(reg_rows)
    return corr_df, reg_df
