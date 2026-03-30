"""Tests for parameter calibration methods."""

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


class TestCalibration(unittest.TestCase):
    """Verify calibration recovers known parameters from synthetic data."""

    def test_gbm_calibration(self):
        gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.08,
            sigma=0.2,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        paths = gbm.simulate(seed=42)
        est = GeometricBrownianMotion.calibrate(paths[0], dt=gbm.dt)
        self.assertAlmostEqual(est["sigma"], 0.2, delta=0.03)
        self.assertAlmostEqual(est["mu"], 0.08, delta=0.15)

    def test_ou_calibration(self):
        ou = OrnsteinUhlenbeck(
            S0=80.0,
            mu=5.0,
            sigma=0.3,
            theta=2.0,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        paths = ou.simulate(seed=42, method="exact")
        est = OrnsteinUhlenbeck.calibrate(paths[0], dt=ou.dt)
        self.assertAlmostEqual(est["theta"], 2.0, delta=0.5)
        self.assertAlmostEqual(est["mu"], 5.0, delta=0.5)
        self.assertAlmostEqual(est["sigma"], 0.3, delta=0.1)

    def test_vasicek_calibration(self):
        v = VasicekModel(
            S0=0.03,
            mu=0.05,
            sigma=0.01,
            a=0.5,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        paths = v.simulate(seed=42, method="exact")
        est = VasicekModel.calibrate(paths[0], dt=v.dt)
        self.assertAlmostEqual(est["a"], 0.5, delta=0.2)
        self.assertAlmostEqual(est["mu"], 0.05, delta=0.02)
        self.assertAlmostEqual(est["sigma"], 0.01, delta=0.005)

    def test_cir_calibration(self):
        cir = CoxIngersollRoss(
            S0=0.04,
            mu=0.05,
            sigma=0.1,
            theta=1.0,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        paths = cir.simulate(seed=42, method="exact")
        est = CoxIngersollRoss.calibrate(paths[0], dt=cir.dt)
        self.assertAlmostEqual(est["theta"], 1.0, delta=0.5)
        self.assertAlmostEqual(est["mu"], 0.05, delta=0.02)
        self.assertAlmostEqual(est["sigma"], 0.1, delta=0.05)

    def test_cev_calibration(self):
        cev = ConstantElasticityOfVariance(
            S0=100.0,
            mu=0.05,
            sigma=0.3,
            gamma=0.5,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        paths = cev.simulate(seed=42)
        est = ConstantElasticityOfVariance.calibrate(paths[0], dt=cev.dt)
        self.assertAlmostEqual(est["gamma"], 0.5, delta=0.3)
        self.assertIsInstance(est["mu"], float)
        self.assertGreater(est["sigma"], 0)

    def test_merton_calibration(self):
        m = MertonJumpDiffusion(
            S0=100.0,
            mu=0.08,
            sigma=0.15,
            lambda_j=5.0,
            mu_j=-0.01,
            sigma_j=0.03,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        paths = m.simulate(seed=42)
        est = MertonJumpDiffusion.calibrate(paths[0], dt=m.dt)
        self.assertGreater(est["sigma"], 0)
        self.assertGreater(est["lambda_j"], 0)
        self.assertGreater(est["sigma_j"], 0)

    def test_variance_gamma_calibration(self):
        vg = VarianceGammaProcess(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            theta=-0.1,
            nu=0.2,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        paths = vg.simulate(seed=42)
        est = VarianceGammaProcess.calibrate(paths[0], dt=vg.dt)
        self.assertAlmostEqual(est["sigma"], 0.2, delta=0.1)
        self.assertGreater(est["nu"], 0)

    def test_heston_calibration(self):
        h = HestonModel(
            S0=100.0,
            v0=0.04,
            mu=0.05,
            sigma=0.3,
            theta=0.04,
            kappa=2.0,
            rho=-0.7,
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        S, v = h.simulate(seed=42)
        est = HestonModel.calibrate(S[0], dt=h.dt)
        self.assertIn("mu", est)
        self.assertIn("sigma", est)
        self.assertIn("v0", est)
        self.assertIn("theta", est)
        self.assertIn("kappa", est)
        self.assertIn("rho", est)
        self.assertGreater(est["sigma"], 0)
        self.assertGreater(est["v0"], 0)
        self.assertTrue(-1 < est["rho"] < 1)

    def test_bates_calibration(self):
        b = BatesModel(
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
            num_paths=1,
            start_date="2015-01-01",
            end_date="2022-12-31",
            granularity="D",
        )
        S, v = b.simulate(seed=42)
        est = BatesModel.calibrate(S[0], dt=b.dt)
        self.assertIn("lambda_j", est)
        self.assertIn("mu_j", est)
        self.assertIn("sigma_j", est)
        self.assertGreater(est["sigma"], 0)
        self.assertGreater(est["lambda_j"], 0)

    # --- Input validation ---

    def test_calibrate_rejects_2d(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion.calibrate(np.ones((2, 10)))

    def test_calibrate_rejects_short(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion.calibrate(np.array([100.0, 101.0]))

    def test_calibrate_rejects_nan(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion.calibrate(np.array([100.0, np.nan, 102.0]))

    def test_calibrate_rejects_negative_prices(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion.calibrate(np.array([100.0, -1.0, 102.0]))

    # --- Return type checks ---

    def test_gbm_returns_dict_with_correct_keys(self):
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        est = GeometricBrownianMotion.calibrate(prices)
        self.assertIsInstance(est, dict)
        self.assertEqual(set(est.keys()), {"mu", "sigma"})

    def test_ou_returns_dict_with_correct_keys(self):
        np.random.seed(0)
        data = np.cumsum(np.random.normal(0, 0.01, 100)) + 5
        est = OrnsteinUhlenbeck.calibrate(data)
        self.assertIsInstance(est, dict)
        self.assertEqual(set(est.keys()), {"mu", "sigma", "theta"})


if __name__ == "__main__":
    unittest.main()
