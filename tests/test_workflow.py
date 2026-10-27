"""Guards for the reusable forecasting workflow and its portable CLI."""
import unittest
from pathlib import Path

import run
from run import build_parser, resolve_data_root
from timedo.presets import get_config


class WorkflowTests(unittest.TestCase):
    def test_refine_validation_snapshot_cli(self):
        p = build_parser()
        args = p.parse_args([
            "--mode", "refine", "--dataset", "Weather", "--pred_len", "192",
            "--latent_mode", "d1", "--checkpoint_policy", "validation",
            "--save_epoch_snapshots", "--skip_test", "--selection_scale", "1.0",
        ])
        self.assertTrue(args.save_epoch_snapshots)
        self.assertTrue(args.skip_test)
        self.assertEqual(args.selection_scale, 1.0)

    def test_published_configs_keep_d1_transition(self):
        for horizon in (96, 192, 336, 720):
            self.assertEqual(get_config("Weather", horizon)["latent_mode"], "d1")
            self.assertEqual(get_config("ETTh1", horizon)["latent_mode"], "d1")
            self.assertEqual(get_config("ETTh2", horizon)["latent_mode"], "d1")

    def test_path_is_not_hard_coded(self):
        target = Path("some/relative/data")
        self.assertEqual(resolve_data_root(target), target.resolve())

    def test_checkpoint_directories_match_project_layout(self):
        root = Path(run.__file__).resolve().parent
        self.assertEqual(run.PRETRAINED_DIR, root / "checkpoints" / "pretrained")
        self.assertEqual(run.INITIALIZATION_DIR, root / "checkpoints" / "initializations")


if __name__ == "__main__":
    unittest.main()
