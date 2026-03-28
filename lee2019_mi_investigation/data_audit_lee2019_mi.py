#!/usr/bin/env python3
"""
Audit Lee2019_MI data: shapes, class balance, sessions, implied sfreq, raw fs from .mat.

Writes JSON + CSV under lee2019_mi_investigation/output/ by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from _common import load_lee2019_mi, repo_root, write_outputs


def _mat_fs(subject: int) -> float:
    from moabb.datasets import Lee2019_MI

    d = Lee2019_MI()
    path = d.data_path(subject)[0]
    m = loadmat(path)
    tr = m["EEG_MI_train"][0, 0]
    return float(tr["fs"].item())


def main() -> int:
    ap = argparse.ArgumentParser(description="Lee2019_MI data audit")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: lee2019_mi_investigation/output)",
    )
    args = ap.parse_args()
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "output")

    from config import get_paradigm

    def paradigm_fn():
        return get_paradigm(dataset="Lee2019_MI")

    X, y, metadata, dataset, paradigm = load_lee2019_mi(args.subjects, paradigm_fn)
    sfreq = float(getattr(paradigm, "resample", None) or 1000.0)
    n_times = X.shape[2]
    implied_duration = n_times / sfreq

    rows = []
    subj_col = metadata["subject"] if "subject" in metadata.columns else None
    for subj in args.subjects:
        if subj_col is not None:
            subj_mask = subj_col.values == subj
        else:
            subj_mask = np.ones(len(metadata), dtype=bool)
        sub_meta = metadata.loc[subj_mask]
        sub_X = X[subj_mask]
        sub_y = y[subj_mask]
        for sess in sorted(sub_meta["session"].unique()):
            sm = sub_meta["session"].values == sess
            yy = sub_y[sm]
            vc = pd.Series(yy).value_counts()
            X_sess = sub_X[sm]
            rows.append(
                {
                    "subject": subj,
                    "session": str(sess),
                    "n_trials": int(len(yy)),
                    "n_channels": X_sess.shape[1],
                    "n_times": X_sess.shape[2],
                    "classes": vc.to_dict(),
                    "balanced": bool(vc.min() == vc.max()) if len(vc) else False,
                }
            )

    summary = {
        "paradigm": str(paradigm),
        "dataset_code": dataset.code,
        "subjects_requested": args.subjects,
        "X_shape": list(X.shape),
        "y_len": len(y),
        "metadata_columns": list(metadata.columns),
        "sessions_globally": sorted(metadata["session"].astype(str).unique().tolist()),
        "sfreq_hz_assumed": sfreq,
        "implied_epoch_duration_s": implied_duration,
        "raw_mat_fs_subject1": _mat_fs(1) if 1 in args.subjects else None,
    }

    write_outputs(out_dir, "data_audit_lee2019_mi", summary, rows)

    detail_path = out_dir / "data_audit_lee2019_mi_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_session_rows": rows,
                "repo_root": str(repo_root()),
            },
            f,
            indent=2,
        )
    print(f"Wrote {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
