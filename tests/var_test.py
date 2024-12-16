import unittest
import numpy as np
from FinStoch.metrics import ValueAtRisk


class TestValueAtRisk(unittest.TestCase):
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
        self.var_calculator = ValueAtRisk(self.simulated_paths)

    def test_calculate_with_alpha(self):
        """
        Test the VaR calculation using alpha.
        """
        alpha = 0.05  # Significance level for 95% confidence
        var = self.var_calculator.calculate(alpha=alpha)
        self.assertTrue(np.isfinite(var), "VaR should be a finite value.")

    def test_calculate_with_confidence_level(self):
        """
        Test the VaR calculation using confidence level.
        """
        confidence_level = 0.95  # 95% confidence
        var = self.var_calculator.calculate(confidence_level=confidence_level)
        self.assertTrue(np.isfinite(var), "VaR should be a finite value.")

    def test_alpha_and_confidence_level_mutual_exclusion(self):
        """
        Test that both alpha and confidence_level cannot be provided simultaneously.
        """
        with self.assertRaises(AssertionError):
            self.var_calculator.calculate(alpha=0.05, confidence_level=0.95)

    def test_no_alpha_or_confidence_level(self):
        """
        Test that an error is raised if neither alpha nor confidence_level is provided.
        """
        with self.assertRaises(AssertionError):
            self.var_calculator.calculate()

    def test_var_value_range(self):
        """
        Test that the calculated VaR is within a reasonable range.
        """
        confidence_level = 0.99  # 99% confidence
        var = self.var_calculator.calculate(confidence_level=confidence_level)
        self.assertLessEqual(var, 0, "VaR should typically be non-positive.")


if __name__ == "__main__":
    unittest.main()
