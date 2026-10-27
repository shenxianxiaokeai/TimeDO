"""Forecasting, teacher isolation, frozen refinement, and shape boundaries."""

import unittest
from dataclasses import replace

import torch

from timedo.config import ModelConfig
from timedo.training.losses import TimeDOLoss
from timedo.models.forecaster import TimeDO


class TimeDOStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.config = ModelConfig(
            seq_len=96, pred_len=24, enc_in=3, period_len=8,
            basis_num=3, time_do_dims="6,3", householder_reflections=3,
        )

    def test_zero_residual_preserves_history_path_and_teacher_cannot_change_prediction(self):
        model = TimeDO(self.config)
        history = torch.randn(2, 96, 3)
        future = torch.randn(2, 24, 3)
        forecast, states = model.forecast_with_states(history, future)
        changed_forecast, changed_states = model.forecast_with_states(history, future + 3.0)
        torch.testing.assert_close(states.predicted_levels[0], states.historical_levels[0], atol=0, rtol=0)
        torch.testing.assert_close(forecast, model(history), atol=0, rtol=0)
        torch.testing.assert_close(forecast, changed_forecast, atol=0, rtol=0)
        self.assertFalse(torch.equal(states.target_levels[0], changed_states.target_levels[0]))
        self.assertTrue(all(not state.requires_grad for state in states.target_levels))

    def test_frozen_refinement_updates_only_transition(self):
        model = TimeDO(self.config)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("transition."))
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
        history, future = torch.randn(2, 96, 3), torch.randn(2, 24, 3)
        prediction, states = model.forecast_with_states(history, future)
        TimeDOLoss(self.config)(prediction, future, states)["loss"].backward()
        optimizer.step()
        changed_transition = False
        for name, parameter in model.named_parameters():
            if name.startswith("transition."):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                changed_transition |= not torch.equal(parameter, before[name])
            else:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        self.assertTrue(changed_transition)

    def test_padding_and_horizon_beyond_history_for_both_normalizations(self):
        for period_norm in (True, False):
            with self.subTest(period_norm=period_norm):
                config = replace(self.config, seq_len=99, pred_len=137, use_period_norm=period_norm)
                model = TimeDO(config)
                history, future = torch.randn(2, 99, 3), torch.randn(2, 137, 3)
                prediction, states = model.forecast_with_states(history, future)
                self.assertEqual(prediction.shape, future.shape)
                torch.testing.assert_close(prediction, model(history), atol=0, rtol=0)
                self.assertTrue(torch.isfinite(TimeDOLoss(config)(prediction, future, states)["loss"]))

    def test_centering_restores_channelwise_translation(self):
        shift = torch.tensor([2.0, -1.0, 3.0]).reshape(1, 1, 3)
        for period_norm in (True, False):
            with self.subTest(period_norm=period_norm):
                model = TimeDO(replace(self.config, use_period_norm=period_norm))
                history = torch.randn(2, 96, 3)
                torch.testing.assert_close(model(history + shift), model(history) + shift, atol=2e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
