"""Streaming regression metrics for memory-bounded evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


class StreamingMetrics:
    """Accumulate element-wise MSE and MAE without storing predictions.

    Inputs may be PyTorch tensors or NumPy-like arrays.  Each update converts
    the error to float32 and accumulates scalar totals in float64, preventing
    evaluation memory from growing with the number of forecast windows.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated errors."""

        self._squared_error = np.float64(0.0)
        self._absolute_error = np.float64(0.0)
        self._count = 0

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Detach a tensor if needed and materialize a CPU NumPy view."""

        if hasattr(value, "detach"):
            value = value.detach().to(device="cpu")
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def update(self, pred: Any, truth: Any) -> None:
        """Add one batch of predictions and targets to the running totals."""

        pred_array = self._to_numpy(pred)
        truth_array = self._to_numpy(truth)
        if pred_array.shape != truth_array.shape:
            raise ValueError(
                f"prediction and target shapes must match; got "
                f"{pred_array.shape} and {truth_array.shape}"
            )
        error = pred_array.astype(np.float32, copy=False) - truth_array.astype(np.float32, copy=False)
        if not np.isfinite(error).all():
            raise ValueError("prediction/target error contains non-finite values")
        self._squared_error += np.sum(np.square(error, dtype=np.float32), dtype=np.float64)
        self._absolute_error += np.sum(np.abs(error, dtype=np.float32), dtype=np.float64)
        self._count += int(error.size)

    def compute(self) -> dict[str, float]:
        """Return mean squared and absolute error for all accumulated values."""

        if self._count == 0:
            raise RuntimeError("cannot compute metrics before at least one update")
        return {
            "mse": float(self._squared_error / self._count),
            "mae": float(self._absolute_error / self._count),
        }


__all__ = ["StreamingMetrics"]
