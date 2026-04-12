"""Tests for multi-asset correlated simulation."""

import unittest

import numpy as np

from FinStoch.multi_asset import CorrelatedGBM


class TestCorrelatedGBM(unittest.TestCase):
    """Verify CorrelatedGBM simulation, analytics, and calibration."""

    def setUp(self):
        self.common = dict(
            S0=[100.0, 50.0, 200.0],
            mu=[0.05, 0.08, 0.03],
            sigma=[0.2, 0.3, 0.15],
            correlation=[
                [1.0, 0.6, 0.3],
                [0.6, 1.0, 0.5],
                [0.3, 0.5, 1.0],
            ],
            num_paths=100,
            start_date="2023-01-01",
            end_date="2024-01-01",
            granularity="D",
        )

    # --- Output shape ---

    def test_simulate_shape(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        self.assertEqual(paths.shape[0], 3)
        self.assertEqual(paths.shape[1], 100)
        self.assertEqual(paths.shape[2], model.num_steps)

    def test_initial_conditions(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        np.testing.assert_array_almost_equal(paths[0, :, 0], 100.0)
        np.testing.assert_array_almost_equal(paths[1, :, 0], 50.0)
        np.testing.assert_array_almost_equal(paths[2, :, 0], 200.0)

    # --- Seed reproducibility ---

    def test_seed_reproducibility(self):
        model = CorrelatedGBM(**self.common)
        a = model.simulate(seed=42)
        b = model.simulate(seed=42)
        np.testing.assert_array_equal(a, b)

    # --- Antithetic ---

    def test_antithetic_shape(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape, (3, 100, model.num_steps))

    def test_antithetic_odd_paths(self):
        model = CorrelatedGBM(**{**self.common, "num_paths": 7})
        paths = model.simulate(seed=42, antithetic=True)
        self.assertEqual(paths.shape[1], 7)

    # --- Correlation recovery ---

    def test_correlation_recovery(self):
        model = CorrelatedGBM(**{**self.common, "num_paths": 10000})
        paths = model.simulate(seed=42)
        realized = model.realized_correlation(paths)
        input_corr = np.array(self.common["correlation"])
        np.testing.assert_allclose(realized, input_corr, atol=0.08)

    def test_identity_correlation_produces_independent(self):
        model = CorrelatedGBM(
            S0=[100.0, 100.0],
            mu=[0.05, 0.05],
            sigma=[0.2, 0.2],
            correlation=[[1.0, 0.0], [0.0, 1.0]],
            num_paths=10000,
            start_date="2023-01-01",
            end_date="2024-01-01",
            granularity="D",
        )
        paths = model.simulate(seed=42)
        realized = model.realized_correlation(paths)
        self.assertAlmostEqual(realized[0, 1], 0.0, delta=0.05)

    # --- Portfolio paths ---

    def test_portfolio_paths_shape(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        portfolio = model.portfolio_paths(paths, weights=[0.5, 0.3, 0.2])
        self.assertEqual(portfolio.shape, (100, model.num_steps))

    def test_portfolio_weights_validation(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        with self.assertRaises(ValueError):
            model.portfolio_paths(paths, weights=[0.5, 0.5])

    # --- Analytics ---

    def test_expected_path_per_asset(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        ep = model.expected_path(paths, asset=0)
        self.assertEqual(ep.shape, (model.num_steps,))

    def test_expected_path_all_assets(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        ep = model.expected_path(paths)
        self.assertEqual(ep.shape, (3, model.num_steps))

    def test_var_cvar(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        portfolio = model.portfolio_paths(paths, weights=[0.5, 0.3, 0.2])
        v = model.var(portfolio, alpha=0.05)
        cv = model.cvar(portfolio, alpha=0.05)
        self.assertIsInstance(v, float)
        self.assertLessEqual(cv, v)

    def test_max_drawdown(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        portfolio = model.portfolio_paths(paths, weights=[0.5, 0.3, 0.2])
        dd = model.max_drawdown(portfolio)
        self.assertEqual(dd.shape, (100,))
        self.assertTrue(np.all(dd >= 0))
        self.assertTrue(np.all(dd <= 1))

    # --- Data conversion ---

    def test_to_dataframe(self):
        model = CorrelatedGBM(**self.common)
        paths = model.simulate(seed=42)
        df = model.to_dataframe(paths, asset=1)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], model.num_steps)

    # --- Calibration ---

    def test_calibrate_round_trip(self):
        model = CorrelatedGBM(**{**self.common, "num_paths": 1})
        paths = model.simulate(seed=42)
        # Transpose to (num_steps, num_assets) for calibration
        data = paths[:, 0, :].T
        est = CorrelatedGBM.calibrate(data, dt=model.dt)
        self.assertEqual(len(est["S0"]), 3)
        self.assertEqual(len(est["mu"]), 3)
        self.assertEqual(len(est["sigma"]), 3)
        self.assertEqual(len(est["correlation"]), 3)

    def test_calibrate_returns_correct_keys(self):
        data = np.random.lognormal(0, 0.01, (100, 2)).cumsum(axis=0)
        est = CorrelatedGBM.calibrate(data)
        self.assertEqual(set(est.keys()), {"S0", "mu", "sigma", "correlation"})

    # --- Input validation ---

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            CorrelatedGBM(
                S0=[100.0, 50.0],
                mu=[0.05],
                sigma=[0.2, 0.3],
                correlation=[[1, 0.5], [0.5, 1]],
                num_paths=10,
                start_date="2023-01-01",
                end_date="2024-01-01",
                granularity="D",
            )

    def test_non_symmetric_correlation_raises(self):
        with self.assertRaises(ValueError):
            CorrelatedGBM(
                S0=[100.0, 50.0],
                mu=[0.05, 0.08],
                sigma=[0.2, 0.3],
                correlation=[[1, 0.5], [0.3, 1]],
                num_paths=10,
                start_date="2023-01-01",
                end_date="2024-01-01",
                granularity="D",
            )

    def test_non_psd_correlation_raises(self):
        with self.assertRaises(ValueError):
            CorrelatedGBM(
                S0=[100.0, 50.0],
                mu=[0.05, 0.08],
                sigma=[0.2, 0.3],
                correlation=[[1, 2], [2, 1]],
                num_paths=10,
                start_date="2023-01-01",
                end_date="2024-01-01",
                granularity="D",
            )

    # --- Properties ---

    def test_properties(self):
        model = CorrelatedGBM(**self.common)
        self.assertEqual(model.num_assets, 3)
        self.assertGreater(model.T, 0)
        self.assertGreater(model.num_steps, 0)
        self.assertGreater(model.dt, 0)


if __name__ == "__main__":
    unittest.main()
