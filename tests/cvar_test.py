import unittest
import numpy as np
from FinStoch.metrics import ExpectedShortfall, ValueAtRisk


class TestExpectedShortfall(unittest.TestCase):
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
        self.es_calculator = ExpectedShortfall(self.simulated_paths)

    def test_calculate_with_alpha(self):
        """
        Test the ES calculation using alpha.
        """
        alpha = 0.05  # Significance level for 95% confidence
        es = self.es_calculator.calculate(alpha=alpha)
        self.assertTrue(np.isfinite(es), "ES should be a finite value.")

    def test_calculate_with_confidence_level(self):
        """
        Test the ES calculation using confidence level.
        """
        confidence_level = 0.95  # 95% confidence
        es = self.es_calculator.calculate(confidence_level=confidence_level)
        self.assertTrue(np.isfinite(es), "ES should be a finite value.")

    def test_alpha_and_confidence_level_mutual_exclusion(self):
        """
        Test that both alpha and confidence_level cannot be provided simultaneously.
        """
        with self.assertRaises(AssertionError):
            self.es_calculator.calculate(alpha=0.05, confidence_level=0.95)

    def test_no_alpha_or_confidence_level(self):
        """
        Test that an error is raised if neither alpha nor confidence_level is provided.
        """
        with self.assertRaises(AssertionError):
            self.es_calculator.calculate()

    def test_es_value_range(self):
        """
        Test that the calculated ES is within a reasonable range.
        """
        confidence_level = 0.99  # 99% confidence
        es = self.es_calculator.calculate(confidence_level=confidence_level)
        self.assertLessEqual(es, 0, "ES should typically be non-positive.")

    def test_es_exceeds_var(self):
        """
        Test that ES is less than or equal to the corresponding VaR.
        """
        confidence_level = 0.95  # 95% confidence
        var_calculator = ValueAtRisk(self.simulated_paths)
        var = var_calculator.calculate(confidence_level=confidence_level)
        es = self.es_calculator.calculate(confidence_level=confidence_level)
        self.assertLessEqual(es, var, "ES should be less than or equal to VaR.")


if __name__ == "__main__":
    unittest.main()
