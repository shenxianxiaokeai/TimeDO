"""Integration and compatibility tests for the model variants."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from timedo.config import ModelConfig
from timedo.models.forecaster import TimeDO
from timedo.presets import DATASETS, HORIZONS, get_config
from timedo.training.engine import _read_checkpoint
from timedo.training.losses import TimeDOLoss


class ModelVariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        self.encoder_only = ModelConfig(
            seq_len=48,
            pred_len=16,
            enc_in=3,
            period_len=4,
            basis_num=3,
            time_do_dims="8,4,3",
            individual=True,
            latent_mode="none",
        )
        self.d1 = replace(self.encoder_only, latent_mode="d1")
        self.history = torch.randn(2, 48, 3)
        self.future = torch.randn(2, 16, 3)

    def test_checkpoint_upgrade_preserves_initial_prediction(self):
        original = TimeDO(self.encoder_only)
        expected = original(self.history)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "s4.pt"
            torch.save(
                {
                    "model_state": original.state_dict(),
                    "model_config": original.config.to_dict(),
                },
                checkpoint,
            )
            upgraded = TimeDO(self.d1)
            _read_checkpoint(
                upgraded,
                checkpoint,
                torch.device("cpu"),
                allow_new_transition=True,
            )

        torch.testing.assert_close(upgraded(self.history), expected, rtol=0, atol=0)
        self.assertTrue(
            all(torch.count_nonzero(parameter) == 0 for parameter in upgraded.transition.parameters())
        )

    def test_d1_teacher_is_detached_and_cannot_change_prediction(self):
        model = TimeDO(self.d1)
        model.transition[0].weight.data.fill_(0.01)

        output, states = model.forecast_with_states(self.history, self.future)
        changed_output, changed_states = model.forecast_with_states(self.history, self.future + 2.0)

        torch.testing.assert_close(output, model(self.history), atol=0, rtol=0)
        torch.testing.assert_close(output, changed_output, atol=0, rtol=0)
        self.assertFalse(torch.equal(states.target_levels[0], changed_states.target_levels[0]))
        self.assertTrue(all(not level.requires_grad for level in states.target_levels))
        self.assertEqual(states.historical_levels[0].shape, (6, 4, 8))
        self.assertEqual(tuple(output.shape), (2, 16, 3))

    def test_frozen_backbone_updates_transition_parameters_only(self):
        model = TimeDO(self.d1)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("transition."))
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01,
        )

        output, states = model.forecast_with_states(self.history, self.future)
        loss = TimeDOLoss(self.d1)(output, self.future, states)["loss"]
        loss.backward()
        optimizer.step()

        changed_transition = False
        for name, parameter in model.named_parameters():
            if name.startswith("transition."):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                changed_transition |= not torch.equal(before[name], parameter)
            else:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        self.assertTrue(changed_transition)

    def test_rank_and_encoder_only_variants(self):
        for mode in ("rank", "none"):
            with self.subTest(mode=mode):
                model = TimeDO(replace(self.d1, latent_mode=mode))
                output, _ = model.forecast_with_states(self.history, self.future)
                self.assertEqual(tuple(output.shape), (2, 16, 3))
                if mode == "none":
                    torch.testing.assert_close(output, model(self.history), atol=0, rtol=0)

    def test_dataset_configs_respect_encoder_depth_limit(self):
        for dataset in DATASETS:
            for horizon in HORIZONS:
                with self.subTest(dataset=dataset, horizon=horizon):
                    config = ModelConfig.from_namespace(get_config(dataset, horizon))
                    self.assertLessEqual(len(TimeDO(config).time_do_dims) - 1, 4)

        with self.assertRaises(ValueError):
            ModelConfig(seq_len=720, period_len=4, basis_num=6, time_do_dims="96,64,48,16,6")


if __name__ == "__main__":
    unittest.main()