"""Tests for the exact transition density scheme."""

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


class TestExactScheme(unittest.TestCase):
    """Verify exact scheme runs and produces correct behavior."""

    def setUp(self):
        self.common = dict(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-03-01",
            granularity="D",
        )

    # --- Category A: Alias processes (exact == euler) ---

    def test_gbm_exact_runs(self):
        model = GeometricBrownianMotion(**self.common)
        paths = model.simulate(seed=42, method="exact")
        self.assertEqual(paths.shape[0], 50)

    def test_gbm_exact_equals_euler(self):
        model = GeometricBrownianMotion(**self.common)
        euler = model.simulate(seed=42, method="euler")
        exact = model.simulate(seed=42, method="exact")
        np.testing.assert_array_equal(euler, exact)

    def test_merton_exact_runs(self):
        model = MertonJumpDiffusion(**self.common, lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)
        paths = model.simulate(seed=42, method="exact")
        self.assertEqual(paths.shape[0], 50)

    def test_merton_exact_equals_euler(self):
        model = MertonJumpDiffusion(**self.common, lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)
        euler = model.simulate(seed=42, method="euler")
        exact = model.simulate(seed=42, method="exact")
        np.testing.assert_array_equal(euler, exact)

    def test_variance_gamma_exact_runs(self):
        model = VarianceGammaProcess(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            theta=-0.1,
            nu=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-03-01",
            granularity="D",
        )
        paths = model.simulate(seed=42, method="exact")
        self.assertEqual(paths.shape[0], 50)

    def test_variance_gamma_exact_equals_euler(self):
        model = VarianceGammaProcess(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            theta=-0.1,
            nu=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-03-01",
            granularity="D",
        )
        euler = model.simulate(seed=42, method="euler")
        exact = model.simulate(seed=42, method="exact")
        np.testing.assert_array_equal(euler, exact)

    # --- Category B: New exact formulas (exact != euler) ---

    def test_ou_exact_runs(self):
        model = OrnsteinUhlenbeck(**self.common, theta=0.5)
        paths = model.simulate(seed=42, method="exact")
        self.assertEqual(paths.shape[0], 50)
        self.assertEqual(paths.shape[1], model.num_steps)

    def test_ou_exact_differs_from_euler(self):
        model = OrnsteinUhlenbeck(**self.common, theta=0.5)
        euler = model.simulate(seed=42, method="euler")
        exact = model.simulate(seed=42, method="exact")
        self.assertFalse(np.array_equal(euler, exact))

    def test_vasicek_exact_runs(self):
        model = VasicekModel(
            S0=0.05,
            mu=0.03,
            sigma=0.01,
            a=0.5,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-03-01",
            granularity="D",
        )
        paths = model.simulate(seed=42, method="exact")
        self.assertEqual(paths.shape[0], 50)

    def test_vasicek_exact_differs_from_euler(self):
        model = VasicekModel(
            S0=0.05,
            mu=0.03,
            sigma=0.01,
            a=0.5,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-03-01",
            granularity="D",
        )
        euler = model.simulate(seed=42, method="euler")
        exact = model.simulate(seed=42, method="exact")
        self.assertFalse(np.array_equal(euler, exact))

    def test_cir_exact_runs(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        paths = model.simulate(seed=42, method="exact")
        self.assertEqual(paths.shape[0], 50)

    def test_cir_exact_differs_from_euler(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        euler = model.simulate(seed=42, method="euler")
        exact = model.simulate(seed=42, method="exact")
        self.assertFalse(np.array_equal(euler, exact))

    def test_cir_exact_non_negative(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        paths = model.simulate(seed=42, method="exact")
        self.assertTrue(np.all(paths >= 0))

    # --- Category C: Warns and falls back to euler ---

    def test_cev_exact_warns_and_falls_back(self):
        model = ConstantElasticityOfVariance(**self.common, gamma=0.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exact = model.simulate(seed=42, method="exact")
            self.assertEqual(len(w), 1)
            self.assertIn("CEV", str(w[0].message))
        euler = model.simulate(seed=42, method="euler")
        np.testing.assert_array_equal(exact, euler)

    def test_heston_exact_warns_and_falls_back(self):
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
            end_date="2023-03-01",
            granularity="D",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            S_exact, v_exact = model.simulate(seed=42, method="exact")
            self.assertEqual(len(w), 1)
            self.assertIn("Heston", str(w[0].message))
        S_euler, v_euler = model.simulate(seed=42, method="euler")
        np.testing.assert_array_equal(S_exact, S_euler)
        np.testing.assert_array_equal(v_exact, v_euler)

    def test_bates_exact_warns_and_falls_back(self):
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
            end_date="2023-03-01",
            granularity="D",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            S_exact, v_exact = model.simulate(seed=42, method="exact")
            self.assertEqual(len(w), 1)
            self.assertIn("Bates", str(w[0].message))
        S_euler, v_euler = model.simulate(seed=42, method="euler")
        np.testing.assert_array_equal(S_exact, S_euler)
        np.testing.assert_array_equal(v_exact, v_euler)

    def test_bootstrap_exact_warns_and_falls_back(self):
        np.random.seed(0)
        prices = 100 * np.exp(np.cumsum(np.concatenate([[0], np.random.normal(0, 0.01, 251)])))
        model = BootstrapMonteCarlo(
            historical_prices=prices,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-03-01",
            granularity="D",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exact = model.simulate(seed=42, method="exact")
            self.assertEqual(len(w), 1)
            self.assertIn("Bootstrap", str(w[0].message))
        euler = model.simulate(seed=42, method="euler")
        np.testing.assert_array_equal(exact, euler)

    # --- Seed reproducibility ---

    def test_exact_seed_reproducibility_ou(self):
        model = OrnsteinUhlenbeck(**self.common, theta=0.5)
        a = model.simulate(seed=42, method="exact")
        b = model.simulate(seed=42, method="exact")
        np.testing.assert_array_equal(a, b)

    def test_exact_seed_reproducibility_vasicek(self):
        model = VasicekModel(
            S0=0.05,
            mu=0.03,
            sigma=0.01,
            a=0.5,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-03-01",
            granularity="D",
        )
        a = model.simulate(seed=42, method="exact")
        b = model.simulate(seed=42, method="exact")
        np.testing.assert_array_equal(a, b)

    def test_exact_seed_reproducibility_cir(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        a = model.simulate(seed=42, method="exact")
        b = model.simulate(seed=42, method="exact")
        np.testing.assert_array_equal(a, b)

    def test_exact_seed_reproducibility_gbm(self):
        model = GeometricBrownianMotion(**self.common)
        a = model.simulate(seed=42, method="exact")
        b = model.simulate(seed=42, method="exact")
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
