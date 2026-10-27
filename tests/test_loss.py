"""Check the published objective, endpoint choices, and stop-gradient target."""

import unittest

import torch

from timedo import ModelConfig, TimeDO, TimeDOLoss
from timedo.training.losses import latent_alignment_loss


class ObjectiveTests(unittest.TestCase):
    def test_weighted_objective_and_gradients(self):
        torch.manual_seed(7)
        config = ModelConfig(seq_len=24, pred_len=12, period_len=2, enc_in=2,
                             basis_num=3, time_do_dims="6,3")
        model = TimeDO(config)
        history = torch.randn(3, 24, 2)
        future = torch.randn(3, 12, 2, requires_grad=True)
        forecast, states = model.forecast_with_states(history, future)
        output = TimeDOLoss(config)(forecast, future.detach(), states)
        final_mse, final_cos = latent_alignment_loss(states.predicted_levels[-1], states.target_levels[-1])
        first_mse, first_cos = latent_alignment_loss(states.predicted_levels[0], states.target_levels[0])
        expected = torch.nn.functional.mse_loss(forecast, future.detach())
        expected = expected + 0.001 * final_mse + 0.0001 * final_cos
        expected = expected + 0.01 * (0.001 * first_mse + 0.0001 * first_cos)
        torch.testing.assert_close(output["loss"], expected)
        output["loss"].backward()
        self.assertIsNone(future.grad)
        self.assertGreater(model.transition.weight.grad.abs().sum().item(), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_latent_training_requires_targets(self):
        config = ModelConfig(seq_len=24, pred_len=12, period_len=2,
                             enc_in=1, basis_num=3, time_do_dims="6,3")
        forecast, states = TimeDO(config).forecast_with_states(torch.randn(2, 24, 1))
        with self.assertRaises(ValueError):
            TimeDOLoss(config)(forecast, torch.randn_like(forecast), states)

    def test_observation_only_ablation(self):
        config = ModelConfig(latent_mode="none", obs_loss_weight=2)
        prediction = torch.tensor([1., 3.], requires_grad=True)
        result = TimeDOLoss(config)(prediction, torch.zeros_like(prediction))
        self.assertEqual(result["loss"].item(), 10.)
        self.assertEqual(result["latent_loss"].item(), 0.)


if __name__ == "__main__":
    unittest.main()
