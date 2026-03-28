#!/usr/bin/env python3
"""
MOABB-protocol-aligned baseline: CSP + LDA with the same paradigm as config.get_paradigm(Lee2019_MI).

- CrossSession: LeaveOneGroupOut on `metadata.session` (same idea as moabb.evaluations.CrossSessionEvaluation).
- Optional --within-session: StratifiedKFold(5) separately for each session (MOABB WithinSession-style).

Outputs JSON summary + CSV rows per subject (and per session for within-session mode).
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
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csp-components", type=int, default=6)
    ap.add_argument(
        "--within-session",
        action="store_true",
        help="Also run WithinSession-style 5-fold CV per session",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "output")

    from config import get_paradigm

    def paradigm_fn():
        return get_paradigm(dataset="Lee2019_MI")

    X, y, metadata, dataset, paradigm = load_lee2019_mi(args.subjects, paradigm_fn)
    y_enc, _ = encode_y(y)

    summary_rows = []
    config = {
        "dataset": dataset.code,
        "paradigm": str(paradigm),
        "subjects": args.subjects,
        "csp_components": args.csp_components,
        "repo_root": str(repo_root()),
        "mode": "cross_session_csp_lda",
    }

    for subj in args.subjects:
        sm = metadata["subject"].values == subj
        Xs = X[sm]
        ys = y_enc[sm]
        meta_s = metadata.loc[sm].reset_index(drop=True)
        aucs, holdout_sess = cross_session_csp_lda_aucs(
            Xs, ys, meta_s, n_components=args.csp_components
        )
        for auc, hs in zip(aucs, holdout_sess):
            summary_rows.append(
                {
                    "subject": subj,
                    "eval": "CrossSession",
                    "holdout_session": hs,
                    "roc_auc": auc,
                }
            )
        summary_rows.append(
            {
                "subject": subj,
                "eval": "CrossSession_mean",
                "holdout_session": "mean",
                "roc_auc": float(np.nanmean(aucs)),
            }
        )

        if args.within_session:
            ws_rows = within_session_per_session_kfold_aucs(
                X,
                y_enc,
                metadata,
                subj,
                n_splits=5,
                random_state=args.seed,
                n_components=args.csp_components,
            )
            for r in ws_rows:
                summary_rows.append(
                    {
                        "subject": subj,
                        "eval": "WithinSession_mean",
                        "holdout_session": r.get("session", ""),
                        "roc_auc": r.get("mean_roc_auc", float("nan")),
                        "extra": str({k: v for k, v in r.items() if k not in ("subject",)}),
                    }
                )

    write_outputs(out_dir, "baseline_moabb_evaluator", config, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
