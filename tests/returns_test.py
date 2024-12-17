import unittest
import numpy as np
from FinStoch.metrics import ExpectedReturn


class TestExpectedReturn(unittest.TestCase):
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
        self.er_calculator = ExpectedReturn(self.simulated_paths)

    def test_calculate_expected_return(self):
        """
        Test the calculation of expected return.
        """
        expected_return = self.er_calculator.calculate()
        self.assertTrue(
            np.isfinite(expected_return), "Expected return should be a finite value."
        )

    def test_expected_return_zero_variation(self):
        """
        Test expected return when there is no variation in simulated paths.
        """
        zero_variation_paths = np.full((1000, 252), 100)
        er_calculator = ExpectedReturn(zero_variation_paths)
        expected_return = er_calculator.calculate()
        self.assertEqual(
            expected_return, 0, "Expected return should be zero for constant paths."
        )


if __name__ == "__main__":
    unittest.main()
