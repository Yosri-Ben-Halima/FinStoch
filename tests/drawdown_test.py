import unittest
import numpy as np
from FinStoch.metrics import MaxDrawdown  # Replace with the correct import path


class TestMaxDrawdown(unittest.TestCase):
    def setUp(self):
        # Example simulated paths for testing
        self.simulated_paths_1 = np.array(
            [[100, 110, 105, 90, 95, 80], [100, 95, 90, 85, 80, 75]]
        )

        self.simulated_paths_2 = np.array(
            [[100, 105, 110, 115, 120], [100, 98, 97, 96, 95]]
        )

        self.simulated_paths_3 = np.array(
            [
                [100, 100, 100, 100],
                [100, 100, 100, 100],
                # No drawdown
            ]
        )

    def test_calculate_max_drawdown(self):
        # Test case 1
        max_drawdown_calculator_1 = MaxDrawdown(self.simulated_paths_1)
        expected_max_drawdown_1 = -0.2727  # Calculation based on example data
        result_1 = max_drawdown_calculator_1.calculate()
        self.assertAlmostEqual(result_1, expected_max_drawdown_1, places=4)

        # Test case 2
        max_drawdown_calculator_2 = MaxDrawdown(self.simulated_paths_2)
        expected_max_drawdown_2 = -0.05  # Calculation based on example data
        result_2 = max_drawdown_calculator_2.calculate()
        self.assertAlmostEqual(result_2, expected_max_drawdown_2, places=4)

    def test_calculate_max_drawdown_constant(self):
        max_drawdown_calculator_3 = MaxDrawdown(self.simulated_paths_3)
        expected_max_drawdown_3 = 0.0  # No drawdown
        result_3 = max_drawdown_calculator_3.calculate()
        self.assertEqual(result_3, expected_max_drawdown_3)


if __name__ == "__main__":
    unittest.main()
