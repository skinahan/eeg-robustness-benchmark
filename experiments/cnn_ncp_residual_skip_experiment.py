#!/usr/bin/env python3
"""
CNN-NCP vs CNN-NCP with residual skip on BNCI2014_001 (subject 1 by default).

- Cross-session evaluation: LeaveOneGroupOut on MOABB `session` (train on N-1 sessions,
  test on the held-out session).
- Perturbation curves: Gaussian, dropout, EOG, AR(1) drift (same augmentors as the main runner).
  EOG uses 10 intensities linearly spaced in [0, 20] (inclusive); other types use ``num_perturb_steps``
  via saturation-based grids (AR(1) uses its own alpha_max grid).
- Outputs per-fold long-form CSV plus aggregated (mean/std across folds) CSV.

See also: plot_cnn_ncp_residual_skip_curves.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from augmentation.noise import EEGNoiseAugmentor
from config import get_paradigm
from evaluation.metrics import compute_classification_metrics
from globals import set_seeds, get_seed
from moabb.datasets import BNCI2014_001
from models.cnnncp import create_cnnncp_classifier, create_cnnncp_residual_skip_classifier
from utils import get_noise_intensities

DATASET = "BNCI2014_001"
DEFAULT_SEED = 42
DEFAULT_SUBJECT = 1
NUM_PERTURB_STEPS = 20
# EOG: fixed grid (not saturation-based); 10 points in [0, 20] inclusive
EOG_INTENSITY_NUM_POINTS = 10
EOG_INTENSITY_MAX = 20.0
EXPERIMENT_MAX_EPOCHS = 100
EXPERIMENT_EARLY_STOPPING = False
PERTURBATION_TYPES: Tuple[str, ...] = ("gaussian", "dropout", "eog", "ar1_drift")
# Match evaluation/unified_experiment_runner.py calibration reproducibility
_CALIBRATION_SEED = 202602
DEFAULT_AR1_RHO = 0.97
DEFAULT_TARGET_SNR_DB = 0.0

DEFAULT_RAW_CSV = _REPO_ROOT / "experiments" / "outputs" / "cnn_ncp_residual_skip_perturb_per_fold.csv"
DEFAULT_AGG_CSV = _REPO_ROOT / "experiments" / "outputs" / "cnn_ncp_residual_skip_perturb_aggregated.csv"
DEFAULT_ROC_PLOT_PATH = (
    _REPO_ROOT / "experiments" / "outputs" / "cnn_ncp_residual_skip_roc_vs_intensity.png"
)

EOG_TEMPLATE_REL = Path("notebooks/eog_mixing_results/generic_eog_mixing_template.npz")


def _eog_template_path() -> str:
    p = _REPO_ROOT / EOG_TEMPLATE_REL
    if p.is_file():
        return str(p)
    return str(EOG_TEMPLATE_REL)


def _compute_alpha_max_ar1(
    X_sample: np.ndarray,
    seed: int = _CALIBRATION_SEED,
    rho: float = DEFAULT_AR1_RHO,
    target_snr_db: float = DEFAULT_TARGET_SNR_DB,
) -> float:
    """SNR-matched alpha_max for AR(1) drift (same construction as UnifiedExperimentRunner)."""
    aug = EEGNoiseAugmentor(
        noise_type="ar1_drift",
        intensity=1.0,
        seed=seed,
        ar1_rho=float(rho),
    )
    X_aug = aug.transform(X_sample)
    eps = X_aug - X_sample
    n_epochs = X_sample.shape[0]
    mean_X_sq = float(np.sum(X_sample**2) / n_epochs)
    mean_eps_sq = float(np.sum(eps**2) / n_epochs)
    if mean_eps_sq <= 0 or not np.isfinite(mean_eps_sq):
        alpha_max_0db = 1.0
    else:
        alpha_max_0db = float(np.sqrt(mean_X_sq / mean_eps_sq))
        if not np.isfinite(alpha_max_0db) or alpha_max_0db <= 0:
            alpha_max_0db = 1.0
    alpha_max = alpha_max_0db * (10.0 ** (-float(target_snr_db) / 20.0))
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        alpha_max = 1.0
    return alpha_max


def _positive_intensity_grid(
    dataset: str,
    noise_type: str,
    num_steps: int,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Strictly positive intensities only; intensity 0 (clean) is emitted separately
    in evaluate_noise_curve for all noise types except EOG (EOG uses a full [0, max] linspace there).
    """
    if noise_type == "eog":
        raise ValueError("EOG intensities are built in evaluate_noise_curve; do not call _positive_intensity_grid")
    if noise_type == "ar1_drift":
        alpha_max = float(_compute_alpha_max_ar1(X_test))
        if num_steps < 1:
            return np.array([], dtype=np.float64)
        return np.linspace(alpha_max / num_steps, alpha_max, num=num_steps, dtype=np.float64)
    grid = np.asarray(
        get_noise_intensities(dataset, noise_type, num_steps=num_steps),
        dtype=np.float64,
    )
    return grid[grid > 1e-12]


def _make_augmentor(
    noise_type: str,
    intensity: float,
    seed: int,
    eog_path: str,
    ar1_rho: float,
) -> EEGNoiseAugmentor:
    kwargs: Dict[str, Any] = {
        "noise_type": noise_type,
        "intensity": float(intensity),
        "seed": int(seed),
    }
    if noise_type == "eog":
        kwargs["eog_template_path"] = eog_path
    if noise_type == "ar1_drift":
        kwargs["ar1_rho"] = float(ar1_rho)
    return EEGNoiseAugmentor(**kwargs)


def load_bnci_with_sessions(
    subject: int = DEFAULT_SUBJECT,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, int, int, int]:
    dataset = BNCI2014_001()
    paradigm = get_paradigm(dataset=DATASET)
    X, y, metadata = paradigm.get_data(dataset, subjects=[subject], return_epochs=False)

    if hasattr(X, "get_data"):
        X = X.get_data()
    X = np.asarray(X, dtype=np.float32)

    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    if "session" not in metadata.columns:
        raise ValueError(
            "MOABB metadata has no 'session' column; cannot run cross-session evaluation."
        )

    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_outputs = len(np.unique(y))
    return X, y, metadata, n_chans, n_times, n_outputs


def evaluate_noise_curve(
    trained_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    noise_type: str,
    dataset: str,
    num_steps: int,
    perturb_seed: int,
    fold_idx: int,
    test_session: str,
    subject: int,
    eog_path: str,
    ar1_rho: float,
) -> pd.DataFrame:
    trained_model.module_.eval()
    rows: List[Dict[str, Any]] = []

    with np.errstate(all="ignore"):
        y_proba_clean = trained_model.predict_proba(X_test)
    m_clean = compute_classification_metrics(y_test, y_proba_clean, num_classes=2)
    clean_roc = float(m_clean.get("roc_auc", 0.0))
    clean_acc = float(m_clean.get("accuracy", 0.0))

    def _append_row(
        intensity: float,
        corrupted_roc: float,
        corrupted_acc: float,
    ) -> None:
        retention = (corrupted_roc / clean_roc * 100.0) if clean_roc > 1e-8 else 0.0
        rel_drop = clean_roc - corrupted_roc
        rows.append(
            {
                "model": model_name,
                "noise_type": noise_type,
                "intensity": float(intensity),
                "clean_roc_auc": clean_roc,
                "clean_accuracy": clean_acc,
                "corrupted_roc_auc": corrupted_roc,
                "corrupted_accuracy": corrupted_acc,
                "retention_pct": retention,
                "relative_drop_roc": rel_drop,
                "fold_idx": fold_idx,
                "test_session": str(test_session),
                "subject": int(subject),
                "dataset": dataset,
            }
        )

    if noise_type == "eog":
        for intensity in np.linspace(0.0, float(EOG_INTENSITY_MAX), num=EOG_INTENSITY_NUM_POINTS):
            if float(intensity) < 1e-12:
                _append_row(0.0, clean_roc, clean_acc)
            else:
                aug = _make_augmentor("eog", float(intensity), perturb_seed, eog_path, ar1_rho)
                X_cor = aug.transform(X_test)
                with np.errstate(all="ignore"):
                    y_proba_c = trained_model.predict_proba(X_cor)
                m_c = compute_classification_metrics(y_test, y_proba_c, num_classes=2)
                _append_row(
                    float(intensity),
                    float(m_c.get("roc_auc", 0.0)),
                    float(m_c.get("accuracy", 0.0)),
                )
        return pd.DataFrame(rows)

    _append_row(0.0, clean_roc, clean_acc)

    positives = _positive_intensity_grid(dataset, noise_type, num_steps, X_test)
    for intensity in positives:
        aug = _make_augmentor(noise_type, float(intensity), perturb_seed, eog_path, ar1_rho)
        X_cor = aug.transform(X_test)
        with np.errstate(all="ignore"):
            y_proba_c = trained_model.predict_proba(X_cor)
        m_c = compute_classification_metrics(y_test, y_proba_c, num_classes=2)
        _append_row(
            float(intensity),
            float(m_c.get("roc_auc", 0.0)),
            float(m_c.get("accuracy", 0.0)),
        )

    return pd.DataFrame(rows)


def aggregate_across_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over folds for each (model, noise_type, intensity)."""
    gcols = ["model", "noise_type", "intensity"]
    agg = (
        df.groupby(gcols, as_index=False)
        .agg(
            clean_roc_auc_mean=("clean_roc_auc", "mean"),
            clean_roc_auc_std=("clean_roc_auc", "std"),
            corrupted_roc_auc_mean=("corrupted_roc_auc", "mean"),
            corrupted_roc_auc_std=("corrupted_roc_auc", "std"),
            clean_accuracy_mean=("clean_accuracy", "mean"),
            corrupted_accuracy_mean=("corrupted_accuracy", "mean"),
            n_folds=("fold_idx", "count"),
        )
    )
    for c in ("clean_roc_auc_std", "corrupted_roc_auc_std"):
        agg[c] = agg[c].fillna(0.0)
    # Aliases for plotting (intensity-0 convention uses these)
    agg["clean_roc_auc"] = agg["clean_roc_auc_mean"]
    agg["corrupted_roc_auc"] = agg["corrupted_roc_auc_mean"]
    return agg


def _save_multi_noise_plot(df_agg: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """Write multi-panel plot and EOG zoom (0–EOG max); return EOG zoom path if written."""
    plot_script = _REPO_ROOT / "experiments" / "plot_cnn_ncp_residual_skip_curves.py"
    spec = importlib.util.spec_from_file_location("_plot_cnn_ncp_residual_skip", plot_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plot module from {plot_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    eog_zoom = out_path.parent / f"{out_path.stem}_eog_zoom_0_{int(EOG_INTENSITY_MAX)}.png"
    ok = mod.plot_all_perturbation_types(
        df_agg,
        out_path,
        eog_zoom_out=eog_zoom,
        eog_zoom_xmax=float(EOG_INTENSITY_MAX),
    )
    return eog_zoom if ok else None


def run_experiment(
    seed: int = DEFAULT_SEED,
    subject: int = DEFAULT_SUBJECT,
    output_raw_csv: Optional[Path] = None,
    output_agg_csv: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    skip_train: bool = False,
    max_epochs: int = EXPERIMENT_MAX_EPOCHS,
    early_stopping: bool = EXPERIMENT_EARLY_STOPPING,
    num_perturb_steps: int = NUM_PERTURB_STEPS,
    noise_types: Sequence[str] = PERTURBATION_TYPES,
    plot_png: Optional[Path] = None,
) -> Dict[str, Any]:
    set_seeds(seed)
    eog_path = _eog_template_path()

    X, y, metadata, n_chans, n_times, n_outputs = load_bnci_with_sessions(subject=subject)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    groups = metadata["session"].values

    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=groups)
    if n_splits < 2:
        raise ValueError(
            f"Cross-session requires >=2 sessions; subject {subject} has {n_splits} split(s). "
            f"Sessions: {np.unique(groups)}"
        )

    print(f"Dataset: {DATASET} subject {subject} | seed {seed}")
    print(f"Cross-session folds: {n_splits} | sessions: {sorted(np.unique(groups).tolist())}")
    print(f"Perturbations: {list(noise_types)} | steps={num_perturb_steps}")
    print(f"Training: max_epochs={max_epochs}, early_stopping={early_stopping}")
    print(f"Shapes: X={X.shape} | {n_chans} ch, {n_times} t, {n_outputs} classes")

    fold_frames: List[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y_enc, groups=groups)):
        test_session = str(metadata.iloc[test_idx[0]]["session"])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        print(
            f"\n=== Fold {fold_idx} | train n={len(y_train)} | test session={test_session!r} n={len(y_test)} ==="
        )

        for name, factory in [
            ("cnn_ncp", create_cnnncp_classifier),
            ("cnn_ncp_residual_skip", create_cnnncp_residual_skip_classifier),
        ]:
            print(f"\n--- {name} ---")
            ckpt: Optional[Path] = None
            if model_dir is not None:
                model_dir.mkdir(parents=True, exist_ok=True)
                ckpt = model_dir / f"{name}_sub{subject}_fold{fold_idx}_seed{seed}.joblib"

            if skip_train and ckpt is not None and ckpt.exists():
                import joblib

                model = joblib.load(ckpt)
                print(f"Loaded checkpoint: {ckpt}")
            else:
                model = factory(
                    n_chans,
                    n_times,
                    n_outputs,
                    max_epochs=max_epochs,
                    early_stopping=early_stopping,
                )
                print("Training...")
                model.fit(X_train, y_train)
                if ckpt is not None:
                    import joblib

                    joblib.dump(model, ckpt)
                    print(f"Saved checkpoint: {ckpt}")

            _ = get_seed()
            y_proba = model.predict_proba(X_test)
            m = compute_classification_metrics(y_test, y_proba, num_classes=n_outputs)
            print(
                f"Clean test ROC-AUC: {m.get('roc_auc', 0.0):.4f} | acc: {m.get('accuracy', 0.0):.4f}"
            )

            for nt in noise_types:
                if nt == "eog" and not Path(eog_path).is_file():
                    print(f"[SKIP] EOG: template missing ({eog_path})")
                    continue
                df_nt = evaluate_noise_curve(
                    model,
                    X_test,
                    y_test,
                    model_name=name,
                    noise_type=nt,
                    dataset=DATASET,
                    num_steps=num_perturb_steps,
                    perturb_seed=seed,
                    fold_idx=fold_idx,
                    test_session=test_session,
                    subject=subject,
                    eog_path=eog_path,
                    ar1_rho=DEFAULT_AR1_RHO,
                )
                fold_frames.append(df_nt)
                last = df_nt.iloc[-1]
                print(
                    f"  {nt}: max intensity {last['intensity']:.4g} -> ROC {last['corrupted_roc_auc']:.4f}"
                )

    combined_raw = pd.concat(fold_frames, ignore_index=True)
    aggregated = aggregate_across_folds(combined_raw)

    if output_raw_csv is not None:
        output_raw_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_raw.to_csv(output_raw_csv, index=False)
        print(f"\nWrote per-fold: {output_raw_csv}")

    if output_agg_csv is not None:
        output_agg_csv.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_agg_csv, index=False)
        print(f"Wrote aggregated: {output_agg_csv}")

    print("\n=== Aggregated clean ROC-AUC (mean over folds, by model) ===")
    clean_by_model = (
        combined_raw.drop_duplicates(subset=["model", "fold_idx"])
        .groupby("model")["clean_roc_auc"]
        .agg(["mean", "std"])
    )
    print(clean_by_model.to_string())

    if plot_png is not None:
        try:
            eog_zoom_path = _save_multi_noise_plot(aggregated, Path(plot_png))
            print(f"\nPlot: {plot_png}")
            if eog_zoom_path is not None:
                print(f"EOG zoom (0–{int(EOG_INTENSITY_MAX)}): {eog_zoom_path}")
        except Exception as e:
            print(f"\n[WARNING] Could not write plot ({plot_png}): {e}")

    return {
        "raw": combined_raw,
        "aggregated": aggregated,
        "clean_by_model": clean_by_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNN-NCP vs residual-skip, cross-session, multi-noise perturbation"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--subject", type=int, default=DEFAULT_SUBJECT)
    parser.add_argument(
        "--output_raw_csv",
        type=str,
        default=str(DEFAULT_RAW_CSV),
        help="Per-fold long-form results",
    )
    parser.add_argument(
        "--output_agg_csv",
        type=str,
        default=str(DEFAULT_AGG_CSV),
        help="Aggregated mean/std across folds",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Directory for per-fold joblib checkpoints",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=EXPERIMENT_MAX_EPOCHS)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--num_perturb_steps", type=int, default=NUM_PERTURB_STEPS)
    parser.add_argument(
        "--noise_types",
        type=str,
        default=",".join(PERTURBATION_TYPES),
        help="Comma-separated: gaussian,dropout,eog,ar1_drift",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=str(DEFAULT_ROC_PLOT_PATH),
        default=None,
        help="Write multi-panel ROC vs intensity PNG",
    )
    args = parser.parse_args()

    allowed_nt = {"gaussian", "dropout", "eog", "ar1_drift"}
    noise_list = tuple(s.strip() for s in args.noise_types.split(",") if s.strip())
    for nt in noise_list:
        if nt not in allowed_nt:
            raise ValueError(f"Unknown noise_type {nt!r}; allowed: {sorted(allowed_nt)}")

    run_experiment(
        seed=args.seed,
        subject=args.subject,
        output_raw_csv=Path(args.output_raw_csv),
        output_agg_csv=Path(args.output_agg_csv),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        skip_train=args.skip_train,
        max_epochs=args.max_epochs,
        early_stopping=args.early_stopping,
        num_perturb_steps=args.num_perturb_steps,
        noise_types=noise_list,
        plot_png=Path(args.plot) if args.plot is not None else None,
    )


if __name__ == "__main__":
    main()
