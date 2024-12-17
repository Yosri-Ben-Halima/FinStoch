import unittest
import numpy as np
from FinStoch.metrics import Volatility


class TestVolatility(unittest.TestCase):
    def setUp(self):
        """
        Set up a simulated paths array for testing.
        """
        np.random.seed(42)  # For reproducibility
        self.simulated_paths = np.hstack(
            (
                np.full((1000, 1), 100),
                np.cumsum(np.random.normal(0, 1, (1000, 252)), axis=1) + 100,
            )
        )
        self.vol_calculator = Volatility(self.simulated_paths)

    def test_calculate_volatility(self):
        """
        Test the calculation of overall volatility.
        """
        volatility = self.vol_calculator.calculate()
        self.assertTrue(np.isfinite(volatility), "Volatility should be a finite value.")

    def test_calculate_downside_volatility(self):
        """
        Test the calculation of downside volatility.
        """
        downside_volatility = self.vol_calculator.downside_vol()
        self.assertTrue(
            np.isfinite(downside_volatility),
            "Downside volatility should be a finite value.",
        )

    def test_calculate_upside_volatility(self):
        """
        Test the calculation of upside volatility.
        """
        upside_volatility = self.vol_calculator.upside_vol()
        self.assertTrue(
            np.isfinite(upside_volatility),
            "Upside volatility should be a finite value.",
        )

    def test_volatility_positive(self):
        """
        Test that the calculated volatilities are non-negative.
        """
        overall_volatility = self.vol_calculator.calculate()
        downside_volatility = self.vol_calculator.downside_vol()
        upside_volatility = self.vol_calculator.upside_vol()
        self.assertGreaterEqual(
            overall_volatility, 0, "Overall volatility should be non-negative."
        )
        self.assertGreaterEqual(
            downside_volatility, 0, "Downside volatility should be non-negative."
        )
        self.assertGreaterEqual(
            upside_volatility, 0, "Upside volatility should be non-negative."
        )

    def test_volatility_with_constant_paths(self):
        """
        Test volatility calculations when paths have no variation.
        """
        constant_paths = np.full((1000, 252), 100)
        vol_calculator = Volatility(constant_paths)
        overall_volatility = vol_calculator.calculate()
        downside_volatility = vol_calculator.downside_vol()
        upside_volatility = vol_calculator.upside_vol()
        self.assertEqual(
            overall_volatility,
            0,
            "Overall volatility should be zero for constant paths.",
        )
        self.assertEqual(
            downside_volatility,
            0,
            "Downside volatility should be zero for constant paths.",
        )
        self.assertEqual(
            upside_volatility, 0, "Upside volatility should be zero for constant paths."
        )


if __name__ == "__main__":
    unittest.main()
