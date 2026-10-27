"""Validated architectural and objective settings for TimeDO.

Dataset-specific training settings live in ``configs/<dataset>.yaml``.  This dataclass only
contains settings that define the forecaster and its training objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from .models.encoder import parse_dimensions


@dataclass(frozen=True)
class ModelConfig:
    """Configure the shared or per-channel Deep Householder forecaster.

    ``time_do_dims`` lists compressed widths after the input segment count;
    the final ``basis_num`` rank may be included as the last entry or omitted.
    The default main method
    corrects the first compressed state (``latent_mode='d1'``); ``rank`` and
    ``none`` are retained solely for controlled ablations.
    """

    seq_len: int = 720
    pred_len: int = 96
    enc_in: int = 7
    period_len: int = 24
    basis_num: int = 6
    time_do_dims: str = "8,6"
    householder_reflections: int = 3
    use_period_norm: bool = True
    individual: bool = False
    latent_mode: str = "d1"
    shallow_alignment: bool = True
    latent_mse_weight: float = 0.001
    latent_cos_weight: float = 0.0001
    shallow_weight: float = 0.01
    obs_loss_weight: float = 1.0
    latent_transition_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("seq_len", "pred_len", "enc_in", "period_len", "basis_num", "householder_reflections"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.latent_mode not in {"d1", "rank", "none"}:
            raise ValueError("latent_mode must be one of: d1, rank, none")
        if len(parse_dimensions(
            (self.seq_len + self.period_len - 1) // self.period_len, self.basis_num, self.time_do_dims
        )) - 1 > 4:
            raise ValueError("Encoder cannot have more than four reduction stages")
        for name in ("latent_mse_weight", "latent_cos_weight", "shallow_weight", "obs_loss_weight"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def from_namespace(cls, settings: Any) -> "ModelConfig":
        """Extract model fields from a CLI namespace or a configuration mapping."""
        if isinstance(settings, cls):
            return settings
        source: Mapping[str, Any] = settings if isinstance(settings, Mapping) else vars(settings)
        values = {field.name: source[field.name] for field in fields(cls) if field.name in source}
        # Archived runs sometimes stored a separate residual sign.  Fold it
        # into the single signed scale; no additional public parameter is needed.
        values["latent_transition_scale"] = float(source.get("latent_residual_sign", 1.0)) * float(
            values.get("latent_transition_scale", cls.latent_transition_scale)
        )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/checkpoint metadata without training-loop state."""
        return asdict(self)


__all__ = ["ModelConfig"]
