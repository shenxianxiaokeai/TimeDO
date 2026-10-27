"""Mathematical guarantees of the Householder hierarchy."""

import unittest

import torch

from timedo.models.encoder import DeepHouseholderEncoder, householder_reflection, parse_dimensions


class HouseholderGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def assert_nested_orthonormal(self, encoder):
        bases = [encoder.projection_matrix(level) for level in range(len(encoder.vectors))]
        for basis in bases:
            identity = torch.eye(basis.shape[1], dtype=basis.dtype)
            torch.testing.assert_close(basis.T @ basis, identity, atol=1e-12, rtol=1e-12)
        for shallow, deep in zip(bases, bases[1:]):
            # Every deeper basis vector is contained in the shallower subspace.
            torch.testing.assert_close(shallow @ shallow.T @ deep, deep, atol=1e-12, rtol=1e-12)

    def test_nested_bases_remain_orthonormal_after_optimizer_steps(self):
        encoder = DeepHouseholderEncoder((12, 8, 5, 3), reflections=3).double()
        self.assert_nested_orthonormal(encoder)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.02)
        value = torch.randn(4, 7, 12, dtype=torch.float64)
        target = torch.randn(4, 7, 3, dtype=torch.float64)
        for _ in range(4):
            optimizer.zero_grad()
            (encoder(value) - target).square().mean().backward()
            optimizer.step()
            self.assert_nested_orthonormal(encoder)

    def test_reflection_is_orthogonal_at_numerical_extremes(self):
        identity = torch.eye(4)
        for scale in (0.0, 1e-30, 1.0, 1e30):
            with self.subTest(scale=scale):
                vector = torch.tensor([1.0, -2.0, 3.0, 4.0]) * scale
                reflection = householder_reflection(identity, vector)
                self.assertTrue(torch.isfinite(reflection).all())
                torch.testing.assert_close(reflection.T @ reflection, identity, atol=1e-6, rtol=1e-6)
                if scale == 0.0:
                    torch.testing.assert_close(reflection, identity, atol=0, rtol=0)

    def test_truncation_is_contractive(self):
        encoder = DeepHouseholderEncoder((12, 7, 3), reflections=3).double()
        value = torch.randn(3, 5, 12, dtype=torch.float64)
        original_norm = torch.linalg.vector_norm(value, dim=-1)
        for state in encoder.forward_levels(value):
            self.assertTrue(torch.all(torch.linalg.vector_norm(state, dim=-1) <= original_norm + 1e-12))

    def test_schedule_rejects_nonadjacent_or_internal_duplicates(self):
        self.assertEqual(parse_dimensions(12, 3, "7,3"), (12, 7, 3))
        self.assertEqual(parse_dimensions(12, 3, "7"), (12, 7, 3))
        for schedule in ("7,3,7", "7,7,3", "7,2"):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                parse_dimensions(12, 3, schedule)


if __name__ == "__main__":
    unittest.main()
