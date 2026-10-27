"""Observation and endpoint-state alignment losses for TimeDO.

The hierarchy is orthogonal by construction, so no orthogonality penalty is
introduced.  Main-method alignment compares the predicted first/final states
with their detached future-window counterparts.  Intermediate states connect
these endpoints through the shared encoder and need no separate loss weights.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from ..models.forecaster import ForecastStates


def latent_alignment_loss(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state MSE and cosine dissimilarity before applying weights.

    MSE averages over every state element.  Cosine similarity flattens the
    period-phase and state-coordinate axes per batch-channel sample, then
    averages across those samples.  Its epsilon follows PyTorch's default;
    it is a numerical safeguard, not an additional tuned hyperparameter.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"state shapes must match: {predicted.shape} versus {target.shape}")
    if predicted.ndim < 2:
        raise ValueError("state tensors need a sample axis and feature axes")
    mse = F.mse_loss(predicted, target)
    cosine = 1.0 - F.cosine_similarity(predicted.flatten(start_dim=1), target.flatten(start_dim=1), dim=-1).mean()
    return mse, cosine


class TimeDOLoss(nn.Module):
    """Compute observation MSE plus weighted final and optional shallow alignment.

    ``total = w_obs * MSE(Y_hat,Y) + alignment(final)
               + shallow_weight * alignment(first)``, where
    ``alignment = latent_mse_weight * state_MSE
                  + latent_cos_weight * cosine_dissimilarity``.

    Shallow alignment is enabled for the main ``d1`` transition.  Disabling
    it, or relocating/disabling the transition, provides explicit ablations.
    Returned raw components make the small coefficients interpretable and
    let callers log both unweighted errors and their contribution to training.
    """

    def __init__(self, config: ModelConfig | Any | None = None) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else ModelConfig.from_namespace(config)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        states: ForecastStates | None = None,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(f"forecast and target shapes differ: {prediction.shape} versus {target.shape}")
        observation = F.mse_loss(prediction, target)
        config = self.config
        result = {
            "loss": config.obs_loss_weight * observation,
            "observation_loss": observation,
            "latent_loss": observation.new_zeros(()),
        }
        if config.latent_mode == "none":
            return result
        if states is None or states.target_levels is None:
            raise ValueError("latent training requires forecast_with_states(history, future)")

        final_mse, final_cos = latent_alignment_loss(states.predicted_levels[-1], states.target_levels[-1])
        latent = config.latent_mse_weight * final_mse + config.latent_cos_weight * final_cos
        result.update(latent_mse=final_mse, latent_cos=final_cos)
        if config.latent_mode == "d1" and config.shallow_alignment:
            shallow_mse, shallow_cos = latent_alignment_loss(states.predicted_levels[0], states.target_levels[0])
            shallow = config.latent_mse_weight * shallow_mse + config.latent_cos_weight * shallow_cos
            latent = latent + config.shallow_weight * shallow
            result.update(shallow_loss=shallow, shallow_mse=shallow_mse, shallow_cos=shallow_cos)
        result["latent_loss"] = latent
        result["loss"] = config.obs_loss_weight * observation + latent
        return result


__all__ = ["TimeDOLoss", "latent_alignment_loss"]
