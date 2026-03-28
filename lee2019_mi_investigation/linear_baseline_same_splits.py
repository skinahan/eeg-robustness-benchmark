#!/usr/bin/env python3
"""
Linear baseline (CSP + LDA) on **CrossSession** splits only — same LeaveOneGroupOut logic as
`unified_experiment_runner` / MOABB `CrossSessionEvaluation` (train on one session, test on the other).

This isolates whether EEGNet underperformance vs this baseline is due to **deep model training**
vs **data separability**. If CSP+LDA is also near chance, prefer preprocessing/window/subject effects.

Outputs match style of `baseline_moabb_evaluator.py` but with a single eval column for clarity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import cross_session_csp_lda_aucs, encode_y, load_lee2019_mi, repo_root, write_outputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
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
        aucs, holdout_sess = cross_session_csp_lda_aucs(
            Xs, ys, meta_s, n_components=args.csp_components
        )
        for auc, hs in zip(aucs, holdout_sess):
            rows.append(
                {
                    "subject": subj,
                    "holdout_session": hs,
                    "roc_auc": auc,
                    "pipeline": "CSP_LDA",
                }
            )
        rows.append(
            {
                "subject": subj,
                "holdout_session": "mean",
                "roc_auc": float(np.nanmean(aucs)),
                "pipeline": "CSP_LDA",
            }
        )

    config = {
        "description": "CrossSession LOGO; same splits as unified runner",
        "dataset": dataset.code,
        "paradigm": str(paradigm),
        "subjects": args.subjects,
        "csp_components": args.csp_components,
        "repo_root": str(repo_root()),
    }
    write_outputs(out_dir, "linear_baseline_same_splits", config, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
