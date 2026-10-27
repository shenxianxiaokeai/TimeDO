"""Validate custom dataset splits, feature order, and train-only scaling."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from timedo.data import LongForecastDataset, build_loaders, infer_channels


class CustomProtocolTests(unittest.TestCase):
    def test_exchange_and_solar_keep_file_feature_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = 1000
            dates = pd.date_range("2020-01-01", periods=rows, freq="h")
            first = np.arange(rows, dtype=np.float64)
            second = first * 2.0 + 5.0
            target = first * 3.0 + 10.0
            frame = pd.DataFrame(
                {"date": dates, "OT": target, "feature_a": first, "feature_b": second}
            )
            frame.to_csv(root / "solar.csv", index=False)

            solar = LongForecastDataset(root, "Solar", "train", 20, 10, data_file="solar.csv")

            self.assertEqual(solar.channels, 3)
            np.testing.assert_allclose(
                solar.scaler_mean,
                [target[:700].mean(), first[:700].mean(), second[:700].mean()],
            )
            self.assertEqual(infer_channels(root, "Solar", "solar.csv"), 3)

    def test_exchange_uses_chronological_row_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = 1000
            values = np.arange(rows, dtype=np.float64)
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=rows, freq="h"),
                    "0": values,
                    "1": values * 2,
                    "OT": values * 3,
                }
            )
            frame.to_csv(root / "exchange.csv", index=False)

            train = LongForecastDataset(root, "Exchange", "train", 20, 10, data_file="exchange.csv")
            validation = LongForecastDataset(root, "Exchange", "val", 20, 10, data_file="exchange.csv")

            self.assertEqual((len(train), len(validation)), (671, 91))
            np.testing.assert_allclose(
                train.scaler_mean,
                [values[:700].mean(), (values[:700] * 2).mean(), (values[:700] * 3).mean()],
            )

    def test_weather_custom_split_uses_training_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = 1200
            raw = np.stack([np.arange(rows), np.arange(rows) * 2.0 + 5.0], axis=1)
            raw[840:] += 1000
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=rows, freq="10min"),
                    "OT": raw[:, 1],
                    "temp": raw[:, 0],
                }
            )
            frame.to_csv(root / "weather.csv", index=False)

            train = LongForecastDataset(root, "Weather", "train", 96, 24)
            validation = LongForecastDataset(root, "Weather", "val", 96, 24)
            test = LongForecastDataset(root, "Weather", "test", 96, 24)

            self.assertEqual(train.channels, 2)
            self.assertEqual((len(train), len(validation), len(test)), (721, 97, 217))
            np.testing.assert_allclose(train.scaler_mean, [raw[:840, 0].mean(), raw[:840, 1].mean()])
            expected = (raw[840:864, [0, 1]] - train.scaler_mean) / train.scaler_scale
            np.testing.assert_allclose(validation[0][1].numpy(), expected)
            self.assertEqual(infer_channels(root, "Weather"), 2)

    def test_weather_row_order_is_preserved_with_repeated_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = 1200
            raw = np.stack(
                [np.arange(rows, dtype=np.float64), 5 + np.arange(rows, dtype=np.float64)],
                axis=1,
            )
            dates = pd.Series(pd.date_range("2020-01-01", periods=rows, freq="10min"))
            dates.iloc[345] = dates.iloc[344]
            dates.iloc[346] = dates.iloc[340]
            pd.DataFrame({"date": dates, "OT": raw[:, 1], "temp": raw[:, 0]}).to_csv(
                root / "weather.csv", index=False
            )

            self.assertEqual(infer_channels(root, "Weather"), 2)
            datasets = build_loaders(root, "Weather", 96, 24, 32)
            train = datasets["train"].dataset

            self.assertEqual(len(train), 721)
            np.testing.assert_allclose(
                train.values[345], (raw[345] - raw[:840].mean(axis=0)) / raw[:840].std(axis=0)
            )
            self.assertEqual(len(datasets["test"].dataset), 217)


if __name__ == "__main__":
    unittest.main()
