"""Metric aggregation across batches of different sizes."""

import unittest

import numpy as np

from timedo.training.metrics import StreamingMetrics


class StreamingMetricsTests(unittest.TestCase):
    def test_metrics_are_element_weighted_across_unequal_batches(self):
        metrics = StreamingMetrics()
        metrics.update(np.array([1.0, 2.0, 3.0]), np.zeros(3))
        metrics.update(np.array([4.0]), np.zeros(1))

        self.assertEqual(metrics.compute(), {"mse": 7.5, "mae": 2.5})


if __name__ == "__main__":
    unittest.main()
