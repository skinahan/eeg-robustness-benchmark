"""
Skorch callback for Hail Mary Block B: batch-level training loss variance per epoch.

Enable via os.environ HAIL_MARY_STABILITY=1 or unified_experiment_runner --hail_mary_stability.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from skorch.callbacks import Callback


class HailMaryStabilityCallback(Callback):
    """Accumulates per-batch train losses within an epoch; logs variance at epoch end."""

    def __init__(self) -> None:
        self._batch_losses: List[float] = []

    def on_epoch_begin(self, net: Any, **kwargs: Any) -> None:
        self._batch_losses = []

    def on_batch_end(self, net: Any, batch: Any = None, training: bool = False, **kwargs: Any) -> None:
        if not training:
            return
        loss = kwargs.get("loss")
        if loss is None:
            for key in ("train_loss", "loss_train", "training_loss"):
                if key in kwargs:
                    loss = kwargs[key]
                    break
        if loss is None:
            return
        try:
            if hasattr(loss, "item"):
                v = float(loss.item())
            else:
                v = float(loss)
            if np.isfinite(v):
                self._batch_losses.append(v)
        except (TypeError, ValueError):
            pass

    def on_epoch_end(self, net: Any, **kwargs: Any) -> None:
        bl = self._batch_losses
        if len(bl) > 1:
            var = float(np.var(bl))
            mean = float(np.mean(bl))
        elif len(bl) == 1:
            var = 0.0
            mean = float(bl[0])
        else:
            var = float("nan")
            mean = float("nan")
        if net.history:
            net.history[-1]["batch_train_loss_var"] = var
            net.history[-1]["batch_train_loss_mean"] = mean
