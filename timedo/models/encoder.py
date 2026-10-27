"""Nested orthogonal reductions used by the Deep Householder encoder.

At stage l, row-vector states are reflected by an orthogonal matrix Q_l and
truncated from d_(l-1) to d_l coordinates.  Writing P_l = Q_l[:, :d_l],
P_l.T @ P_l = I.  Products of these semi-orthogonal maps therefore retain
orthonormal columns while defining nested subspaces in the original input.
Orthogonality is structural and requires no penalty in the loss.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def householder_reflection(value: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply ``x -> x - 2 <x,u> u``, where ``u = v / ||v||``.

    A zero vector defines the identity transformation.  Very small vectors
    are rescaled before normalization if their norm underflows; clamping a
    nonzero norm to an arbitrary epsilon would break orthogonality.  The
    ordinary path preserves the arithmetic of the archived checkpoints.
    """
    if vector.ndim != 1 or value.shape[-1] != vector.numel():
        raise ValueError("reflection vector must match the final input dimension")
    norm = vector.norm()
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    unit = vector / safe_norm
    maximum = vector.abs().amax()
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    scaled = vector / safe_maximum
    scaled_norm = scaled.norm()
    stable_unit = scaled / torch.where(scaled_norm > 0, scaled_norm, torch.ones_like(scaled_norm))
    ordinary = (norm > torch.finfo(vector.dtype).tiny) & torch.isfinite(norm)
    unit = torch.where(ordinary, unit, stable_unit)
    projection = (value * unit).sum(dim=-1, keepdim=True)
    return value - 2.0 * projection * unit


def parse_dimensions(segment_count: int, rank: int, hidden: str | Iterable[int]) -> tuple[int, ...]:
    """Build a strictly descending ``segment_count -> hidden... -> rank`` schedule.

    A repeated final endpoint is collapsed: ``30, 8, 6, 6`` becomes ``30, 8, 6``.
    Other repeated widths are rejected.  This keeps a schedule such as ``'8,6'`` convenient when
    the final rank is also six, as in the reference configurations.
    """
    if isinstance(hidden, str):
        values = [int(item.strip()) for item in hidden.split(",") if item.strip()]
    else:
        values = [int(item) for item in hidden]
    widths = [int(segment_count), *values]
    if widths[-1] != int(rank):
        widths.append(int(rank))
    dimensions = tuple(widths)
    if len(dimensions) < 2 or any(item < 1 for item in dimensions):
        raise ValueError(f"encoder needs at least one positive-width reduction: {dimensions}")
    if any(a <= b for a, b in zip(dimensions, dimensions[1:])):
        raise ValueError(f"encoder dimensions must strictly descend: {dimensions}")
    return dimensions


class DeepHouseholderEncoder(nn.Module):
    """Progressively compress period profiles into nested low-rank states.

    Inputs have shape ``(..., segment_count)``.  Each stage applies a small
    product of Householder reflections and keeps its first output-width
    coordinates.  The final rank and all intermediate states use the same
    learned hierarchy for history, predicted states, and future targets.
    """

    def __init__(self, dimensions: Sequence[int], reflections: int = 1) -> None:
        super().__init__()
        self.dimensions = tuple(int(width) for width in dimensions)
        if len(self.dimensions) < 2 or any(width < 1 for width in self.dimensions):
            raise ValueError("dimensions must contain at least two positive widths")
        if any(a <= b for a, b in zip(self.dimensions, self.dimensions[1:])):
            raise ValueError(f"dimensions must strictly descend: {self.dimensions}")
        self.reflections = int(reflections)
        if self.reflections < 1:
            raise ValueError("reflections must be positive")
        # Keep this name/layout for direct loading of archived encoder tensors.
        self.vectors = nn.ParameterList([
            nn.Parameter(torch.randn(self.reflections, width) * 0.02)
            for width in self.dimensions[:-1]
        ])

    def _stage(self, value: torch.Tensor, stage: int) -> torch.Tensor:
        if value.shape[-1] != self.dimensions[stage]:
            raise ValueError(f"stage {stage} expects width {self.dimensions[stage]}, got {value.shape[-1]}")
        for vector in self.vectors[stage]:
            value = householder_reflection(value, vector)
        return value[..., :self.dimensions[stage + 1]]

    def forward_levels(self, value: torch.Tensor) -> list[torch.Tensor]:
        """Return every compressed representation, ordered shallow to deep."""
        levels = []
        for stage in range(len(self.vectors)):
            value = self._stage(value, stage)
            levels.append(value)
        return levels

    def encode_first(self, value: torch.Tensor) -> torch.Tensor:
        """Encode only the shallow state, before the latent transition."""
        return self._stage(value, 0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the final low-rank representation."""
        return self.forward_levels(value)[-1]

    def reduce_levels_from(self, value: torch.Tensor, start_stage: int) -> list[torch.Tensor]:
        """Continue the shared hierarchy after an already compressed state."""
        if not 0 <= start_stage <= len(self.vectors):
            raise ValueError("start_stage is outside the encoder hierarchy")
        levels = []
        for stage in range(start_stage, len(self.vectors)):
            value = self._stage(value, stage)
            levels.append(value)
        return levels

    def reduce_from(self, value: torch.Tensor, start_stage: int) -> torch.Tensor:
        """Return the final state after continuing from ``start_stage``."""
        levels = self.reduce_levels_from(value, start_stage)
        return levels[-1] if levels else value

    def projection_matrix(self, level: int = -1) -> torch.Tensor:
        """Return the input-space semi-orthogonal basis for one encoded level."""
        identity = torch.eye(self.dimensions[0], device=self.vectors[0].device, dtype=self.vectors[0].dtype)
        return self.forward_levels(identity)[level]

    @torch.no_grad()
    def orthogonality_error(self) -> torch.Tensor:
        """Report mean squared error of the deepest basis Gram matrix from I."""
        basis = self.projection_matrix()
        gram = basis.transpose(0, 1) @ basis
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - identity).square().mean()


__all__ = ["DeepHouseholderEncoder", "householder_reflection", "parse_dimensions"]
