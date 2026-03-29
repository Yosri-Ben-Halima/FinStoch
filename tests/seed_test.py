"""Tests for seed-controlled reproducibility across all processes."""

import unittest

import numpy as np

from FinStoch.processes import (
    GeometricBrownianMotion,
    MertonJumpDiffusion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    ConstantElasticityOfVariance,
    HestonModel,
)


class TestSeedReproducibility(unittest.TestCase):
    """Verify that simulate(seed=N) produces identical results on repeat calls."""

    def setUp(self):
        self.common = dict(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=10,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )

    def test_gbm_seed(self):
        model = GeometricBrownianMotion(**self.common)
        a = model.simulate(seed=42)
        b = model.simulate(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_gbm_different_seeds(self):
        model = GeometricBrownianMotion(**self.common)
        a = model.simulate(seed=42)
        b = model.simulate(seed=99)
        self.assertFalse(np.array_equal(a, b))

    def test_gbm_no_seed(self):
        model = GeometricBrownianMotion(**self.common)
        a = model.simulate()
        b = model.simulate()
        # Without seed, results should differ (with overwhelming probability)
        self.assertFalse(np.array_equal(a, b))

    def test_merton_seed(self):
        model = MertonJumpDiffusion(**self.common, lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)
        a = model.simulate(seed=42)
        b = model.simulate(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_ou_seed(self):
        model = OrnsteinUhlenbeck(**self.common, theta=0.5)
        a = model.simulate(seed=42)
        b = model.simulate(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_cir_seed(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        a = model.simulate(seed=42)
        b = model.simulate(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_cev_seed(self):
        model = ConstantElasticityOfVariance(**self.common, gamma=0.5)
        a = model.simulate(seed=42)
        b = model.simulate(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_heston_seed(self):
        model = HestonModel(
            S0=100.0,
            v0=0.04,
            mu=0.05,
            sigma=0.3,
            theta=0.04,
            kappa=2.0,
            rho=-0.7,
            num_paths=10,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        s1, v1 = model.simulate(seed=42)
        s2, v2 = model.simulate(seed=42)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(v1, v2)


if __name__ == "__main__":
    unittest.main()
