"""Minimal observation decoder shared across variables and period phases."""

from torch import nn


class LinearDecoder(nn.Linear):
    """Map the final latent rank directly to forecast-period coordinates.

    For a state tensor ``(batch * channels, period_len, rank)``, this affine
    layer returns ``(batch * channels, period_len, forecast_segments)``.
    TimeDO then restores temporal order and trims to the forecast horizon.
    No hidden decoder layers or nonlinearities are introduced.
    """

    def __init__(self, rank: int, forecast_segments: int) -> None:
        if int(rank) < 1 or int(forecast_segments) < 1:
            raise ValueError("decoder rank and forecast_segments must be positive")
        super().__init__(int(rank), int(forecast_segments))


__all__ = ["LinearDecoder"]
