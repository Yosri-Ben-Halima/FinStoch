"""Tests for analytics methods and to_dataframe on StochasticProcess."""

import unittest

import numpy as np
import pandas as pd

from FinStoch.processes import GeometricBrownianMotion, HestonModel


class TestToDataFrame(unittest.TestCase):
    """Tests for the to_dataframe() method."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=5,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_dataframe(self):
        df = self.gbm.to_dataframe(self.paths)
        self.assertIsInstance(df, pd.DataFrame)

    def test_shape(self):
        df = self.gbm.to_dataframe(self.paths)
        self.assertEqual(df.shape, self.paths.shape)

    def test_columns_are_datetimeindex(self):
        df = self.gbm.to_dataframe(self.paths)
        self.assertIsInstance(df.columns, pd.DatetimeIndex)

    def test_values_match(self):
        df = self.gbm.to_dataframe(self.paths)
        np.testing.assert_array_equal(df.values, self.paths)

    def test_heston_price(self):
        heston = HestonModel(
            S0=100.0,
            v0=0.04,
            mu=0.05,
            sigma=0.3,
            theta=0.04,
            kappa=2.0,
            rho=-0.7,
            num_paths=5,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        result = heston.simulate(seed=42)
        df = heston.to_dataframe(result, variance=False)
        np.testing.assert_array_equal(df.values, result[0])

    def test_heston_variance(self):
        heston = HestonModel(
            S0=100.0,
            v0=0.04,
            mu=0.05,
            sigma=0.3,
            theta=0.04,
            kappa=2.0,
            rho=-0.7,
            num_paths=5,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        result = heston.simulate(seed=42)
        df = heston.to_dataframe(result, variance=True)
        np.testing.assert_array_equal(df.values, result[1])


class TestSummaryStatistics(unittest.TestCase):
    """Tests for the summary_statistics() method."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=100,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_dict(self):
        result = self.gbm.summary_statistics(self.paths)
        self.assertIsInstance(result, dict)

    def test_keys(self):
        result = self.gbm.summary_statistics(self.paths)
        expected_keys = {"mean", "std", "skew", "kurtosis", "min", "max"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_shapes(self):
        result = self.gbm.summary_statistics(self.paths)
        for key, val in result.items():
            self.assertEqual(len(val), self.paths.shape[1], msg=f"{key} length mismatch")

    def test_mean_values(self):
        result = self.gbm.summary_statistics(self.paths)
        expected_mean = np.mean(self.paths, axis=0)
        np.testing.assert_array_almost_equal(result["mean"], expected_mean)

    def test_min_max(self):
        result = self.gbm.summary_statistics(self.paths)
        self.assertTrue(np.all(result["min"] <= result["mean"]))
        self.assertTrue(np.all(result["max"] >= result["mean"]))

    def test_initial_values(self):
        result = self.gbm.summary_statistics(self.paths)
        # At t=0, all paths start at S0=100, so std=0, skew=0
        self.assertAlmostEqual(result["mean"][0], 100.0)
        self.assertAlmostEqual(result["std"][0], 0.0)


class TestConfidenceBands(unittest.TestCase):
    """Tests for the confidence_bands() method."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=1000,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_tuple(self):
        result = self.gbm.confidence_bands(self.paths)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_shapes(self):
        lower, upper = self.gbm.confidence_bands(self.paths)
        self.assertEqual(len(lower), self.paths.shape[1])
        self.assertEqual(len(upper), self.paths.shape[1])

    def test_lower_below_upper(self):
        lower, upper = self.gbm.confidence_bands(self.paths)
        self.assertTrue(np.all(lower <= upper))

    def test_initial_bands_at_s0(self):
        lower, upper = self.gbm.confidence_bands(self.paths)
        # At t=0 all paths are S0=100
        self.assertAlmostEqual(lower[0], 100.0)
        self.assertAlmostEqual(upper[0], 100.0)

    def test_custom_level(self):
        lower_90, upper_90 = self.gbm.confidence_bands(self.paths, level=0.90)
        lower_99, upper_99 = self.gbm.confidence_bands(self.paths, level=0.99)
        # 99% bands should be wider than 90% bands at non-initial steps
        self.assertTrue(np.all(lower_99[-1] <= lower_90[-1]))
        self.assertTrue(np.all(upper_99[-1] >= upper_90[-1]))


class TestExpectedPath(unittest.TestCase):
    """Tests for the expected_path() method."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_1d_array(self):
        result = self.gbm.expected_path(self.paths)
        self.assertEqual(result.ndim, 1)

    def test_shape(self):
        result = self.gbm.expected_path(self.paths)
        self.assertEqual(len(result), self.paths.shape[1])

    def test_equals_mean(self):
        result = self.gbm.expected_path(self.paths)
        np.testing.assert_array_equal(result, np.mean(self.paths, axis=0))

    def test_initial_value(self):
        result = self.gbm.expected_path(self.paths)
        self.assertAlmostEqual(result[0], 100.0)


class TestMedianPath(unittest.TestCase):
    """Tests for the median_path() method."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_1d_array(self):
        result = self.gbm.median_path(self.paths)
        self.assertEqual(result.ndim, 1)

    def test_shape(self):
        result = self.gbm.median_path(self.paths)
        self.assertEqual(len(result), self.paths.shape[1])

    def test_equals_median(self):
        result = self.gbm.median_path(self.paths)
        np.testing.assert_array_equal(result, np.median(self.paths, axis=0))

    def test_initial_value(self):
        result = self.gbm.median_path(self.paths)
        self.assertAlmostEqual(result[0], 100.0)

    def test_median_between_min_and_max(self):
        result = self.gbm.median_path(self.paths)
        self.assertTrue(np.all(result >= np.min(self.paths, axis=0)))
        self.assertTrue(np.all(result <= np.max(self.paths, axis=0)))


class TestVaR(unittest.TestCase):
    """Tests for the var() method (Value at Risk)."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=1000,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_float(self):
        result = self.gbm.var(self.paths)
        self.assertIsInstance(result, float)

    def test_var_below_mean(self):
        var_val = self.gbm.var(self.paths, alpha=0.05)
        mean_terminal = np.mean(self.paths[:, -1])
        self.assertLess(var_val, mean_terminal)

    def test_higher_alpha_gives_higher_var(self):
        var_1 = self.gbm.var(self.paths, alpha=0.01)
        var_5 = self.gbm.var(self.paths, alpha=0.05)
        # 1% VaR should be lower (worse) than 5% VaR
        self.assertLess(var_1, var_5)

    def test_var_within_range(self):
        var_val = self.gbm.var(self.paths, alpha=0.05)
        self.assertGreaterEqual(var_val, np.min(self.paths[:, -1]))
        self.assertLessEqual(var_val, np.max(self.paths[:, -1]))


class TestCVaR(unittest.TestCase):
    """Tests for the cvar() method (Conditional Value at Risk)."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=1000,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_float(self):
        result = self.gbm.cvar(self.paths)
        self.assertIsInstance(result, float)

    def test_cvar_below_var(self):
        var_val = self.gbm.var(self.paths, alpha=0.05)
        cvar_val = self.gbm.cvar(self.paths, alpha=0.05)
        # CVaR (expected shortfall) should be <= VaR
        self.assertLessEqual(cvar_val, var_val)

    def test_cvar_within_range(self):
        cvar_val = self.gbm.cvar(self.paths, alpha=0.05)
        self.assertGreaterEqual(cvar_val, np.min(self.paths[:, -1]))


class TestMaxDrawdown(unittest.TestCase):
    """Tests for the max_drawdown() method."""

    def setUp(self):
        self.gbm = GeometricBrownianMotion(
            S0=100.0,
            mu=0.05,
            sigma=0.2,
            num_paths=50,
            start_date="2023-01-01",
            end_date="2023-01-10",
            granularity="D",
        )
        self.paths = self.gbm.simulate(seed=42)

    def test_returns_1d_array(self):
        result = self.gbm.max_drawdown(self.paths)
        self.assertEqual(result.ndim, 1)

    def test_shape(self):
        result = self.gbm.max_drawdown(self.paths)
        self.assertEqual(len(result), self.paths.shape[0])

    def test_values_between_0_and_1(self):
        result = self.gbm.max_drawdown(self.paths)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_known_drawdown(self):
        # Manually create paths with known drawdown
        paths = np.array([[100, 120, 90, 110]])  # peak=120, trough=90 -> dd=0.25
        result = self.gbm.max_drawdown(paths)
        self.assertAlmostEqual(result[0], 0.25)

    def test_monotonically_increasing_path(self):
        # No drawdown on a strictly increasing path
        paths = np.array([[100, 110, 120, 130]])
        result = self.gbm.max_drawdown(paths)
        self.assertAlmostEqual(result[0], 0.0)


if __name__ == "__main__":
    unittest.main()
