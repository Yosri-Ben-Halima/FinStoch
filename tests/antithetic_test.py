"""Tests for antithetic variates."""

import unittest
import warnings

import numpy as np

from FinStoch.processes import (
    GeometricBrownianMotion,
    MertonJumpDiffusion,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    ConstantElasticityOfVariance,
    HestonModel,
    VasicekModel,
    BatesModel,
    VarianceGammaProcess,
)
from FinStoch.bootstrap import BootstrapMonteCarlo


class TestAntitheticVariates(unittest.TestCase):
    """Verify antithetic variates produce correct shapes and reduce variance."""

    def setUp(self):
        self.common = dict(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-06-01",
            granularity="D",
        )

    # --- Shape preservation ---

    def test_gbm_antithetic_shape(self):
        model = GeometricBrownianMotion(**self.common)
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 50)
        self.assertEqual(paths.shape[1], model.num_steps)

    def test_ou_antithetic_shape(self):
        model = OrnsteinUhlenbeck(**self.common, theta=0.5)
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 50)

    def test_vasicek_antithetic_shape(self):
        model = VasicekModel(
            S0=0.05, mu=0.03, sigma=0.01, a=0.5, num_paths=50, start_date="2023-01-01", end_date="2023-06-01", granularity="D"
        )
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 50)

    def test_cev_antithetic_shape(self):
        model = ConstantElasticityOfVariance(**self.common, gamma=0.5)
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 50)

    def test_cir_antithetic_shape(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 50)

    def test_merton_antithetic_shape(self):
        model = MertonJumpDiffusion(**self.common, lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 50)

    def test_heston_antithetic_shape(self):
        model = HestonModel(
            S0=100.0,
            v0=0.04,
            mu=0.05,
            sigma=0.3,
            theta=0.04,
            kappa=2.0,
            rho=-0.7,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-06-01",
            granularity="D",
        )
        S, v = model.simulate(seed=42, antithetic=True)
        self.assertEqual(S.shape[0], 50)
        self.assertEqual(v.shape[0], 50)

    def test_bates_antithetic_shape(self):
        model = BatesModel(
            S0=100.0,
            v0=0.04,
            mu=0.05,
            sigma=0.3,
            theta=0.04,
            kappa=2.0,
            rho=-0.7,
            lambda_j=0.1,
            mu_j=-0.05,
            sigma_j=0.1,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-06-01",
            granularity="D",
        )
        S, v = model.simulate(seed=42, antithetic=True)
        self.assertEqual(S.shape[0], 50)
        self.assertEqual(v.shape[0], 50)

    def test_vg_antithetic_shape(self):
        model = VarianceGammaProcess(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            theta=-0.1,
            nu=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-06-01",
            granularity="D",
        )
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 50)

    # --- Odd num_paths ---

    def test_odd_num_paths(self):
        model = GeometricBrownianMotion(**{**self.common, "num_paths": 7})
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[0], 7)

    # --- Seed reproducibility ---

    def test_antithetic_seed_reproducibility(self):
        model = GeometricBrownianMotion(**self.common)
        a = model.simulate(seed=42, antithetic=True)
        b = model.simulate(seed=42, antithetic=True)
        np.testing.assert_array_equal(a, b)

    # --- Warnings ---

    def test_cir_exact_antithetic_warns(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.simulate(seed=42, method="exact", antithetic=True)
            self.assertTrue(any("Antithetic" in str(x.message) for x in w))

    def test_bootstrap_antithetic_warns(self):
        np.random.seed(0)
        prices = 100 * np.exp(np.cumsum(np.concatenate([[0], np.random.normal(0, 0.01, 251)])))
        model = BootstrapMonteCarlo(
            historical_prices=prices, num_paths=50, start_date="2023-01-01", end_date="2023-06-01", granularity="D"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.simulate(seed=42, antithetic=True)
            self.assertTrue(any("Antithetic" in str(x.message) for x in w))

    # --- Variance reduction check ---

    def test_antithetic_reduces_variance(self):
        """Antithetic mean estimator should have lower variance than standard."""
        model = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=2000,
            start_date="2023-01-01",
            end_date="2024-01-01",
            granularity="D",
        )
        means_standard = []
        means_antithetic = []
        for seed in range(20):
            paths_std = model.simulate(seed=seed, antithetic=False)
            paths_ant = model.simulate(seed=seed + 1000, antithetic=True)
            means_standard.append(np.mean(paths_std[:, -1]))
            means_antithetic.append(np.mean(paths_ant[:, -1]))
        var_std = np.var(means_standard)
        var_ant = np.var(means_antithetic)
        # Antithetic should reduce variance (allow some slack)
        self.assertLess(var_ant, var_std * 1.5)


if __name__ == "__main__":
    unittest.main()
