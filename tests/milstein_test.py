"""Tests for the Milstein discretization scheme."""

import unittest

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


class TestMilsteinScheme(unittest.TestCase):
    """Verify Milstein scheme runs and produces correct behavior."""

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

    def test_gbm_milstein_runs(self):
        model = GeometricBrownianMotion(**self.common)
        paths = model.simulate(seed=42, method="milstein")
        self.assertEqual(paths.shape[0], 50)

    def test_gbm_milstein_differs_from_euler(self):
        model = GeometricBrownianMotion(**self.common)
        euler = model.simulate(seed=42, method="euler")
        milstein = model.simulate(seed=42, method="milstein")
        self.assertFalse(np.array_equal(euler, milstein))

    def test_merton_milstein_runs(self):
        model = MertonJumpDiffusion(**self.common, lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)
        paths = model.simulate(seed=42, method="milstein")
        self.assertEqual(paths.shape[0], 50)

    def test_merton_milstein_differs_from_euler(self):
        model = MertonJumpDiffusion(**self.common, lambda_j=0.1, mu_j=-0.05, sigma_j=0.1)
        euler = model.simulate(seed=42, method="euler")
        milstein = model.simulate(seed=42, method="milstein")
        self.assertFalse(np.array_equal(euler, milstein))

    def test_ou_milstein_equals_euler(self):
        model = OrnsteinUhlenbeck(**self.common, theta=0.5)
        euler = model.simulate(seed=42, method="euler")
        milstein = model.simulate(seed=42, method="milstein")
        np.testing.assert_array_equal(euler, milstein)

    def test_vasicek_milstein_equals_euler(self):
        model = VasicekModel(
            S0=0.05, mu=0.03, sigma=0.01, a=0.5, num_paths=50, start_date="2023-01-01", end_date="2023-03-01", granularity="D"
        )
        euler = model.simulate(seed=42, method="euler")
        milstein = model.simulate(seed=42, method="milstein")
        np.testing.assert_array_equal(euler, milstein)

    def test_cir_milstein_runs(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        paths = model.simulate(seed=42, method="milstein")
        self.assertEqual(paths.shape[0], 50)

    def test_cir_milstein_differs_from_euler(self):
        model = CoxIngersollRoss(**self.common, theta=0.5)
        euler = model.simulate(seed=42, method="euler")
        milstein = model.simulate(seed=42, method="milstein")
        self.assertFalse(np.array_equal(euler, milstein))

    def test_cev_milstein_runs(self):
        model = ConstantElasticityOfVariance(**self.common, gamma=0.5)
        paths = model.simulate(seed=42, method="milstein")
        self.assertEqual(paths.shape[0], 50)

    def test_cev_milstein_differs_from_euler(self):
        model = ConstantElasticityOfVariance(**self.common, gamma=0.5)
        euler = model.simulate(seed=42, method="euler")
        milstein = model.simulate(seed=42, method="milstein")
        self.assertFalse(np.array_equal(euler, milstein))

    def test_heston_milstein_runs(self):
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
        S, v = model.simulate(seed=42, method="milstein")
        self.assertEqual(S.shape[0], 50)

    def test_heston_milstein_differs_from_euler(self):
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
        _, v_euler = model.simulate(seed=42, method="euler")
        _, v_milstein = model.simulate(seed=42, method="milstein")
        self.assertFalse(np.array_equal(v_euler, v_milstein))

    def test_bates_milstein_runs(self):
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
        S, v = model.simulate(seed=42, method="milstein")
        self.assertEqual(S.shape[0], 50)

    def test_variance_gamma_milstein_raises(self):
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
        with self.assertRaises(ValueError):
            model.simulate(method="milstein")

    def test_invalid_method_raises(self):
        model = GeometricBrownianMotion(**self.common)
        with self.assertRaises(ValueError):
            model.simulate(method="runge_kutta")

    def test_milstein_seed_reproducibility(self):
        model = GeometricBrownianMotion(**self.common)
        a = model.simulate(seed=42, method="milstein")
        b = model.simulate(seed=42, method="milstein")
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
