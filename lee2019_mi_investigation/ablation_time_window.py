#!/usr/bin/env python3
"""
Ablate MotorImagery time window for Lee2019_MI: same CSP+LDA + CrossSession LOGO, different (tmin, tmax).

Default windows:
  - paper_1_3p5: (1.0, 3.5)  — paper online window (MOABB comment)
  - full_0_4:    (0.0, 4.0) — explicit full task length
  - mid_0p5_3p5: (0.5, 3.5) — intermediate

Optional --include-moabb-default: tmin=0, tmax=None (dataset interval [0,4]).

Outputs CSV with per-fold and mean ROC-AUC per subject per window.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from _common import (
    cross_session_csp_lda_aucs,
    encode_y,
    get_paradigm_for_window,
    load_lee2019_mi,
    repo_root,
    write_outputs,
)


def _windows(
    include_moabb_default: bool,
) -> List[Tuple[str, float, Optional[float]]]:
    w = [
        ("paper_1_3p5", 1.0, 3.5),
        ("full_0_4", 0.0, 4.0),
        ("mid_0p5_3p5", 0.5, 3.5),
    ]
    if include_moabb_default:
        w.append(("moabb_tmax_none", 0.0, None))
    return w


def _paradigm_factory(
    tmin: float,
    tmax: Optional[float],
    resample: float,
) -> Callable[[], object]:
    if tmax is None:

        def factory():
            from moabb.paradigms import MotorImagery

            return MotorImagery(
                events=["left_hand", "right_hand"],
                fmin=8,
                fmax=30,
                tmin=0.0,
                tmax=None,
                baseline=None,
                resample=resample,
                n_classes=2,
            )

        return factory

    def factory(tm=tmin, tx=tmax, rs=resample):
        return get_paradigm_for_window(tm, tx, resample=rs)

    return factory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--csp-components", type=int, default=6)
    ap.add_argument("--include-moabb-default", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (Path(__file__).resolve().parent / "output")

    from config import get_dataset_sampling_rate

    resample = float(get_dataset_sampling_rate("Lee2019_MI"))
    rows = []
    windows = _windows(args.include_moabb_default)
    last_dataset_code = "Lee2019-MI"

    for wname, tmin, tmax in windows:
        paradigm_fn = _paradigm_factory(tmin, tmax, resample)
        X, y, metadata, dataset, paradigm = load_lee2019_mi(args.subjects, paradigm_fn)
        last_dataset_code = dataset.code
        y_enc, _ = encode_y(y)
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
                        "window_name": wname,
                        "tmin": tmin,
                        "tmax": tmax if tmax is not None else "None",
                        "subject": subj,
                        "holdout_session": hs,
                        "roc_auc": auc,
                        "X_shape": str(Xs.shape),
                        "paradigm": str(paradigm),
                    }
                )
            rows.append(
                {
                    "window_name": wname,
                    "tmin": tmin,
                    "tmax": tmax if tmax is not None else "None",
                    "subject": subj,
                    "holdout_session": "mean",
                    "roc_auc": float(np.nanmean(aucs)),
                    "X_shape": str(Xs.shape),
                    "paradigm": str(paradigm),
                }
            )

    config = {
        "windows": [str(w) for w in windows],
        "subjects": args.subjects,
        "csp_components": args.csp_components,
        "repo_root": str(repo_root()),
        "dataset": last_dataset_code,
    }
    write_outputs(out_dir, "ablation_time_window", config, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
