"""
Skorch callback for Hail Mary Block B: loss volatility for stability summaries.

Within-epoch batch variance:
  Reads per-batch `train_loss` / `valid_loss` from `net.history[-1]['batches']`
  (same as NeuralNet.run_single_epoch `record_batch`).

When there is only one training batch per epoch (common when the train split fits in one
batch at the configured batch size), within-epoch spread is reported as **0.0** (same as numpy
``np.var([x])``) so logs and JSON avoid NaN; interpret with ``n_train_batches_recorded``.
For cross-topology stability comparisons, prefer epoch-to-epoch metrics
(``train_loss_epoch_abs_delta``, ``valid_loss_epoch_to_epoch_var``, rolling std) when batch variance is always zero.

Epoch-level metrics use **0.0** on the first epoch (no prior epoch to compare)
and **0.0** rolling std until at least two epochs exist.

Enable via os.environ HAIL_MARY_STABILITY=1 or unified_experiment_runner --hail_mary_stability.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
from skorch.callbacks import Callback


def _var_mean(xs: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        # Single batch: no within-epoch spread; use 0.0 like np.var([x]) (not NaN).
        return 0.0, float(arr[0])
    return float(np.var(arr)), float(np.mean(arr))


class HailMaryStabilityCallback(Callback):
    """Logs batch_train_loss_var / batch_valid_loss_var from skorch history batches."""

    def __init__(self) -> None:
        self._batch_losses_fallback: List[float] = []

    def on_epoch_begin(self, net: Any, **kwargs: Any) -> None:
        self._batch_losses_fallback = []

    def on_batch_end(self, net: Any, batch: Any = None, training: bool = False, **kwargs: Any) -> None:
        if not training:
            return
        loss = kwargs.get("loss")
        if loss is None:
            return
        try:
            v = float(loss.item()) if hasattr(loss, "item") else float(loss)
            if np.isfinite(v):
                self._batch_losses_fallback.append(v)
        except (TypeError, ValueError):
            pass

    def on_epoch_end(self, net: Any, **kwargs: Any) -> None:
        train_losses: List[float] = []
        valid_losses: List[float] = []
        n_batches = 0
        try:
            if net.history:
                row = net.history[-1]
                batches = row.get("batches") or []
                n_batches = len(batches)
                for b in batches:
                    if not isinstance(b, dict):
                        continue
                    if "train_loss" in b:
                        train_losses.append(float(b["train_loss"]))
                    if "valid_loss" in b:
                        valid_losses.append(float(b["valid_loss"]))
        except Exception:
            pass

        if len(train_losses) < 1 and self._batch_losses_fallback:
            train_losses = list(self._batch_losses_fallback)

        tr_var, tr_mean = _var_mean(train_losses)
        va_var, va_mean = _var_mean(valid_losses)

        if net.history:
            net.history[-1]["batch_train_loss_var"] = tr_var
            net.history[-1]["batch_train_loss_mean"] = tr_mean
            net.history[-1]["n_skorch_batches_total"] = float(n_batches)
            net.history[-1]["n_train_batches_recorded"] = float(len(train_losses))
            if valid_losses:
                net.history[-1]["batch_valid_loss_var"] = va_var
                net.history[-1]["batch_valid_loss_mean"] = va_mean

            # Epoch-level volatility (useful when only 1 train batch/epoch → batch var is nan)
            cur_t = net.history[-1].get("train_loss")
            if cur_t is None and np.isfinite(tr_mean):
                cur_t = tr_mean
            cur_v = net.history[-1].get("valid_loss")
            if cur_v is None and valid_losses:
                cur_v = float(np.mean(valid_losses))
            try:
                cur_t = float(cur_t) if cur_t is not None else float("nan")
            except (TypeError, ValueError):
                cur_t = float("nan")
            try:
                cur_v = float(cur_v) if cur_v is not None else float("nan")
            except (TypeError, ValueError):
                cur_v = float("nan")

            tl_series: List[float] = []
            for i in range(max(0, len(net.history) - 1)):
                row = net.history[i]
                v = row.get("train_loss")
                if v is not None and np.isfinite(v):
                    tl_series.append(float(v))
            if np.isfinite(cur_t):
                tl_series.append(cur_t)

            vl_series: List[float] = []
            for i in range(max(0, len(net.history) - 1)):
                row = net.history[i]
                v = row.get("valid_loss")
                if v is not None and np.isfinite(v):
                    vl_series.append(float(v))
            if np.isfinite(cur_v):
                vl_series.append(cur_v)

            if len(tl_series) == 0:
                net.history[-1]["train_loss_epoch_abs_delta"] = float("nan")
                net.history[-1]["train_loss_epoch_roll_std_5"] = float("nan")
            elif len(tl_series) == 1:
                net.history[-1]["train_loss_epoch_abs_delta"] = 0.0
                net.history[-1]["train_loss_epoch_roll_std_5"] = 0.0
            else:
                net.history[-1]["train_loss_epoch_abs_delta"] = float(abs(tl_series[-1] - tl_series[-2]))
                w = min(5, len(tl_series))
                net.history[-1]["train_loss_epoch_roll_std_5"] = float(np.std(tl_series[-w:]))

            if len(vl_series) == 0:
                net.history[-1]["valid_loss_epoch_abs_delta"] = float("nan")
                net.history[-1]["valid_loss_epoch_roll_std_5"] = float("nan")
            elif len(vl_series) == 1:
                net.history[-1]["valid_loss_epoch_abs_delta"] = 0.0
                net.history[-1]["valid_loss_epoch_roll_std_5"] = 0.0
            else:
                net.history[-1]["valid_loss_epoch_abs_delta"] = float(abs(vl_series[-1] - vl_series[-2]))
                wv = min(5, len(vl_series))
                net.history[-1]["valid_loss_epoch_roll_std_5"] = float(np.std(vl_series[-wv:]))
