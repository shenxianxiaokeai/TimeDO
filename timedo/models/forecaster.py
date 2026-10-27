"""TimeDO: Deep Orthogonality with a future-aligned latent state transition.

The inference path is history -> period profiles -> first Householder state
-> latent transition -> remaining shared reductions -> linear decoder.
During training only, a future-shifted window is encoded by the same hierarchy
to provide detached shallow and final latent targets.  Future observations
never enter the forecasting branch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from ..config import ModelConfig
from .decoder import LinearDecoder
from .encoder import DeepHouseholderEncoder, parse_dimensions
from .transition import LatentTransition


@dataclass(frozen=True)
class ForecastStates:
    """State tensors exposed to the objective, without embedding losses in the model.

    Each tuple is ordered from the first compressed representation to the
    final rank.  ``historical_levels`` describes X, ``predicted_levels`` the
    future-state forecasting path, and ``target_levels`` the detached X+
    teacher (absent for inference).  Intermediate predicted states feed the
    next reduction even though alignment is imposed only at the endpoints.
    """

    historical_levels: tuple[torch.Tensor, ...]
    predicted_levels: tuple[torch.Tensor, ...]
    target_levels: tuple[torch.Tensor, ...] | None = None

    @property
    def historical_first(self) -> torch.Tensor:
        return self.historical_levels[0]


class TimeDO(nn.Module):
    """A shared hierarchical orthogonal forecaster with a minimal linear head.

    ``forward(history)`` is the only inference API and accepts no future
    observations.  ``forecast_with_states(history, future)`` additionally
    constructs detached training targets; ``TimeDOLoss`` owns the objective.
    ``latent_mode='none'`` disables transition/alignment for an encoder-only
    ablation, while ``'rank'`` moves the transition to the final state.
    """

    method_name = "TimeDO"

    def __init__(self, config: ModelConfig | Any) -> None:
        super().__init__()
        self.config = ModelConfig.from_namespace(config)
        config = self.config
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.enc_in = int(config.enc_in)
        self.period_len = int(config.period_len)
        self.rank = int(config.basis_num)
        self.use_period_norm = bool(config.use_period_norm)
        self.individual = bool(config.individual)
        self.latent_mode = config.latent_mode
        self.shallow_alignment = bool(config.shallow_alignment)
        self.segment_count = (self.seq_len + self.period_len - 1) // self.period_len
        self.forecast_segments = (self.pred_len + self.period_len - 1) // self.period_len
        self.padding = self.segment_count * self.period_len - self.seq_len
        self.time_do_dims = parse_dimensions(self.segment_count, self.rank, config.time_do_dims)
        self.householder_reflections = int(config.householder_reflections)
        width = self.time_do_dims[1] if self.latent_mode == "d1" else self.rank
        self.transition_width = width if self.latent_mode != "none" else 0

        # These discarded Linear initializations intentionally consume the
        # same random draws as the original reference construction.  They
        # add no parameters, modules, or operations to the forecasting graph.
        # This keeps archived seeded training runs reproducible after refactoring.
        if self.individual:
            _discarded_encoders = [nn.Linear(self.segment_count, self.rank) for _ in range(self.enc_in)]
            self.decoder = nn.ModuleList([
                LinearDecoder(self.rank, self.forecast_segments) for _ in range(self.enc_in)
            ])
            self.encoder = nn.ModuleList([
                DeepHouseholderEncoder(self.time_do_dims, self.householder_reflections)
                for _ in range(self.enc_in)
            ])
            if self.latent_mode != "none":
                self.transition = nn.ModuleList([
                    LatentTransition(width, config.latent_transition_scale)
                    for _ in range(self.enc_in)
                ])
        else:
            _discarded_encoder = nn.Linear(self.segment_count, self.rank)
            self.decoder = LinearDecoder(self.rank, self.forecast_segments)
            self.encoder = DeepHouseholderEncoder(self.time_do_dims, self.householder_reflections)
            if self.latent_mode != "none":
                self.transition = LatentTransition(width, config.latent_transition_scale)

    @property
    def latent_transition_scale(self) -> float:
        """Return the fixed residual scale recorded in model metadata."""
        return float(self.config.latent_transition_scale)

    def set_transition_scale(self, scale: float) -> None:
        """Update the fixed scale while retaining every learned weight."""
        if self.latent_mode == "none":
            raise ValueError("the encoder-only ablation has no latent transition")
        if self.individual:
            for transition in self.transition:
                transition.set_scale(scale)
        else:
            self.transition.set_scale(scale)
        self.config = replace(self.config, latent_transition_scale=float(scale))

    def _validate_history(self, history: torch.Tensor) -> None:
        if history.ndim != 3:
            raise ValueError("history must have shape (batch, seq_len, channels)")
        if history.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected history (*, {self.seq_len}, {self.enc_in}), got {tuple(history.shape)}"
            )
        if not history.is_floating_point():
            raise ValueError("history must be a floating-point tensor")

    def _period_profiles(self, history: torch.Tensor) -> torch.Tensor:
        """Rearrange (B,L,C) into (B*C, period_len, number_of_periods).

        Each row follows the same phase across successive periods.  Partial
        periods are padded with a repeated preceding period tail, preserving
        the reference behavior for non-multiple history lengths.
        """
        batch, _, channels = history.shape
        value = history.permute(0, 2, 1)
        if self.padding:
            start = (self.segment_count - 1) * self.period_len
            indices = torch.arange(start - self.padding, start, device=value.device) % self.seq_len
            value = torch.cat([value, value.index_select(-1, indices)], dim=-1)
        value = value.reshape(batch, channels, self.segment_count, self.period_len)
        return value.permute(0, 1, 3, 2).reshape(-1, self.period_len, self.segment_count)

    def _normalise(self, profiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Center each period phase, or center the whole channel in an ablation."""
        if self.use_period_norm:
            mean = profiles.mean(dim=-1, keepdim=True)
        else:
            mean = profiles.reshape(profiles.shape[0], -1).mean(dim=-1).reshape(-1, 1, 1)
        return profiles - mean, mean

    def _restore_prediction(self, decoded: torch.Tensor, mean: torch.Tensor, batch: int) -> torch.Tensor:
        """Undo history centering, restore temporal order, and trim padding."""
        decoded = decoded + mean
        value = decoded.reshape(batch, self.enc_in, self.period_len, self.forecast_segments)
        value = value.permute(0, 1, 3, 2).reshape(batch, self.enc_in, -1).permute(0, 2, 1)
        return value[:, :self.pred_len]

    def _forecast_states(self, profiles: torch.Tensor, channel: int | None = None) -> ForecastStates:
        """Compute aligned states for a shared encoder or one independent channel."""
        encoder = self.encoder if channel is None else self.encoder[channel]
        transition = getattr(self, "transition", None)
        if channel is not None and transition is not None:
            transition = transition[channel]
        historical = tuple(encoder.forward_levels(profiles))
        if self.latent_mode == "d1":
            first = transition(historical[0])
            predicted = (first, *encoder.reduce_levels_from(first, start_stage=1))
        elif self.latent_mode == "rank":
            predicted = (*historical[:-1], transition(historical[-1]))
        else:
            predicted = historical
        return ForecastStates(historical, tuple(predicted))

    def forecast_with_states(
        self, history: torch.Tensor, future: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ForecastStates]:
        """Forecast from history and optionally encode a detached future teacher.

        X+ is the final ``seq_len`` rows of ``[history, future]``.  When the
        horizon does not exceed the history, this is ``[X[:, T:], Y]``.
        Both windows use centering statistics from X, so their states share
        coordinates.  The future branch is always stop-gradient, even during
        joint encoder refinement.
        """
        self._validate_history(history)
        batch = history.shape[0]
        profiles, mean = self._normalise(self._period_profiles(history))
        if future is not None:
            expected = (batch, self.pred_len, self.enc_in)
            if tuple(future.shape) != expected:
                raise ValueError(f"expected future shape {expected}, got {tuple(future.shape)}")
            if future.device != history.device or future.dtype != history.dtype:
                raise ValueError("history and future must share device and dtype")
            shifted = torch.cat([history, future], dim=1)[:, -self.seq_len:]
            future_profiles = self._period_profiles(shifted) - mean
        else:
            future_profiles = None
        if not self.individual:
            states = self._forecast_states(profiles)
            prediction = self._restore_prediction(self.decoder(states.predicted_levels[-1]), mean, batch)
            if future_profiles is not None:
                with torch.no_grad():
                    targets = tuple(self.encoder.forward_levels(future_profiles))
                states = ForecastStates(states.historical_levels, states.predicted_levels, targets)
            return prediction, states

        # Keep the flattened B*C order: batch-major then channel.  Each
        # independent channel is encoded (and teacher-encoded) by that same
        # channel's own encoder.  Never leak future observations into forecasts.
        channel_profiles = profiles.reshape(batch, self.enc_in, self.period_len, self.segment_count)
        channel_future = (future_profiles.reshape(batch, self.enc_in, self.period_len, self.segment_count)
                          if future_profiles is not None else None)
        individual_states = []
        decoded = []
        for channel in range(self.enc_in):
            states = self._forecast_states(channel_profiles[:, channel], channel)
            decoded.append(self.decoder[channel](states.predicted_levels[-1]))
            if channel_future is not None:
                with torch.no_grad():
                    targets = tuple(self.encoder[channel].forward_levels(channel_future[:, channel]))
                states = ForecastStates(states.historical_levels, states.predicted_levels, targets)
            individual_states.append(states)
        merged = ForecastStates(
            historical_levels=tuple(torch.stack([state.historical_levels[i] for state in individual_states], dim=1).reshape(batch * self.enc_in, self.period_len, -1)
                                    for i in range(len(individual_states[0].historical_levels))),
            predicted_levels=tuple(torch.stack([state.predicted_levels[i] for state in individual_states], dim=1).reshape(batch * self.enc_in, self.period_len, -1)
                                   for i in range(len(individual_states[0].predicted_levels))),
            target_levels=(tuple(torch.stack([state.target_levels[i] for state in individual_states], dim=1).reshape(batch * self.enc_in, self.period_len, -1)
                                 for i in range(len(individual_states[0].predicted_levels)))
                           if channel_future is not None else None),
        )
        decoded_all = torch.stack(decoded, dim=1).reshape(-1, self.period_len, self.forecast_segments)
        return self._restore_prediction(decoded_all, mean, batch), merged

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return forecasts using history alone; no future target is required."""
        self._validate_history(history)
        batch = history.shape[0]
        profiles, mean = self._normalise(self._period_profiles(history))
        if not self.individual:
            # The main inference graph visits every encoder stage once.  In
            # particular it does not compute a historical deep state that
            # would be discarded after correcting the shallow representation.
            if self.latent_mode == "d1":
                first = self.transition(self.encoder.encode_first(profiles))
                final = self.encoder.reduce_from(first, start_stage=1)
            elif self.latent_mode == "rank":
                final = self.transition(self.encoder(profiles))
            else:
                final = self.encoder(profiles)
            return self._restore_prediction(self.decoder(final), mean, batch)
        if self.latent_mode != "none":
            return self.forecast_with_states(history)[0]
        profiles = profiles.reshape(batch, self.enc_in, self.period_len, self.segment_count)
        decoded = torch.stack([
            self.decoder[channel](self.encoder[channel](profiles[:, channel]))
            for channel in range(self.enc_in)
        ], dim=1).reshape(-1, self.period_len, self.forecast_segments)
        return self._restore_prediction(decoded, mean, batch)

    def diagnostics(self) -> dict[str, object]:
        """Return structural diagnostics; these values do not alter the loss."""
        encoders = list(self.encoder) if self.individual else [self.encoder]
        errors = torch.stack([encoder.orthogonality_error() for encoder in encoders])
        return {
            "dimensions": self.time_do_dims,
            "rank": self.rank,
            "orthogonality_error": errors.mean(),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


__all__ = ["TimeDO", "ForecastStates"]
