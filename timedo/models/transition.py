"""A single affine residual correction in the first compressed state space."""

from __future__ import annotations

import math

import torch
from torch import nn


class LatentTransition(nn.Linear):
    """Predict a future-aligned state as ``z + scale * (W z + b)``.

    The only trainable parameters are the square affine map ``W`` and ``b``.
    ``scale`` is a fixed signed hyperparameter; it adds no trainable scalar.
    Zero initialization preserves the input state exactly before refinement.
    Subclassing ``nn.Linear`` retains simple ``transition.weight/bias`` keys
    when an archived checkpoint is loaded into the refactored model.
    """

    def __init__(self, width: int, scale: float = 1.0) -> None:
        if int(width) < 1:
            raise ValueError("transition width must be positive")
        super().__init__(int(width), int(width))
        self.set_scale(scale)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def set_scale(self, scale: float) -> None:
        """Set the non-trainable residual scale, e.g. for checkpoint evaluation."""
        if not math.isfinite(float(scale)):
            raise ValueError("transition scale must be finite")
        self.scale = float(scale)

    def correction(self, state: torch.Tensor) -> torch.Tensor:
        """Return the unscaled affine residual ``W z + b``."""
        return super().forward(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return the corrected state; scale is applied exactly once."""
        return state + self.scale * self.correction(state)


__all__ = ["LatentTransition"]
