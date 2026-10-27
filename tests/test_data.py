"""ETT split boundaries and training-only scaling."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from timedo.data import ETTHourDataset


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="timedo_data_test_")
        cls.root = Path(cls.directory.name)
        steps = np.arange(14400, dtype=np.float64)
        # A large held-out offset makes a scaler fitted beyond training fail.
        cls.raw = np.stack([steps, steps * 0.3 + 2], axis=1)
        cls.raw[8640:] += 100000
        frame = pd.DataFrame(cls.raw, columns=["a", "b"])
        frame["date"] = pd.date_range("2016-01-01", periods=len(frame), freq="h")
        frame.to_csv(cls.root / "ETTh1.csv", index=False)

    @classmethod
    def tearDownClass(cls):
        # TemporaryDirectory owns this exact verified temporary folder only.
        assert cls.root.resolve().name.startswith("timedo_data_test_")
        cls.directory.cleanup()

    def test_scaler_uses_training_rows_only(self):
        dataset = ETTHourDataset(self.root, "ETTh1", "train", 96, 24)
        np.testing.assert_allclose(dataset.scaler_mean, self.raw[:8640].mean(axis=0))
        np.testing.assert_allclose(dataset.scaler_scale, self.raw[:8640].std(axis=0))
        self.assertEqual(dataset[0][0].dtype.__str__(), "torch.float64")

    def test_held_out_targets_start_at_split_boundaries(self):
        for split, start, count in (("val", 8640, 2880), ("test", 11520, 2880)):
            with self.subTest(split=split):
                dataset = ETTHourDataset(self.root, "ETTh1", split, 96, 24)
                history, target = dataset[0]
                expected = (self.raw[start:start + 24] - dataset.scaler_mean) / dataset.scaler_scale
                np.testing.assert_allclose(target.numpy(), expected)
                self.assertEqual(tuple(history.shape), (96, 2))
                self.assertEqual(len(dataset), count - 24 + 1)


if __name__ == "__main__":
    unittest.main()
