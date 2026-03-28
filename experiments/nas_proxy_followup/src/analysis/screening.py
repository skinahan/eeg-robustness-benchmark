"""Top-k screening vs random baseline."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def simulate_topk_screening(
    df: pd.DataFrame,
    proxy_column: str,
    target_column: str,
    q_values: List[float],
    num_random_trials: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Rank by proxy (higher better), take top q% by count, compare mean target to random subsets.
    """
    rng = np.random.default_rng(seed)
    sub = df[[proxy_column, target_column]].dropna()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values(proxy_column, ascending=False).reset_index(drop=True)
    n = len(sub)
    rows = []
    for q in q_values:
        k = max(1, int(np.ceil(n * q)))
        top_mean = float(sub[target_column].iloc[:k].mean())
        random_means = []
        for _ in range(num_random_trials):
            idx = rng.choice(n, size=k, replace=False)
            random_means.append(float(sub[target_column].iloc[idx].mean()))
        rows.append(
            {
                "q": q,
                "k": k,
                "topk_mean_target": top_mean,
                "random_mean_mean": float(np.mean(random_means)),
                "random_mean_std": float(np.std(random_means)),
            }
        )
    return pd.DataFrame(rows)
