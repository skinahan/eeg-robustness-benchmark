#!/usr/bin/env python3
"""
Compare CrossSession vs WithinSession ROC-AUC (CSP+LDA) on Lee2019_MI with repo paradigm.

- CrossSession: LeaveOneGroupOut on session; report mean ROC-AUC across folds.
- WithinSession: For each session separately, StratifiedKFold(5); report mean ROC-AUC per session.

Use this to see whether poor CrossSession scores are due to **session transfer** vs **within-session** difficulty.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (
    cross_session_csp_lda_aucs,
    encode_y,
    load_lee2019_mi,
    repo_root,
    within_session_per_session_kfold_aucs,
    write_outputs,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csp-components", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "output")

    from config import get_paradigm

    def paradigm_fn():
        return get_paradigm(dataset="Lee2019_MI")

    X, y, metadata, dataset, paradigm = load_lee2019_mi(args.subjects, paradigm_fn)
    y_enc, _ = encode_y(y)

    rows = []
    for subj in args.subjects:
        sm = metadata["subject"].values == subj
        Xs = X[sm]
        ys = y_enc[sm]
        meta_s = metadata.loc[sm].reset_index(drop=True)
        aucs, _ = cross_session_csp_lda_aucs(
            Xs, ys, meta_s, n_components=args.csp_components
        )
        rows.append(
            {
                "subject": subj,
                "eval": "CrossSession_LOGO_mean",
                "roc_auc": float(np.nanmean(aucs)),
                "fold_aucs": str(aucs),
            }
        )
        ws = within_session_per_session_kfold_aucs(
            X,
            y_enc,
            metadata,
            subj,
            n_splits=5,
            random_state=args.seed,
            n_components=args.csp_components,
        )
        for r in ws:
            rows.append(
                {
                    "subject": subj,
                    "eval": f"WithinSession_sess_{r['session']}",
                    "roc_auc": r.get("mean_roc_auc", float("nan")),
                    "fold_aucs": "",
                }
            )

    config = {
        "dataset": dataset.code,
        "paradigm": str(paradigm),
        "subjects": args.subjects,
        "repo_root": str(repo_root()),
    }
    write_outputs(out_dir, "split_comparison", config, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
