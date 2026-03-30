import unittest
import numpy as np
from FinStoch.bootstrap import BootstrapMonteCarlo


class TestBootstrapMonteCarlo(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        # Synthetic historical prices: 252 days of GBM-like data
        returns = np.random.normal(0.0002, 0.01, 251)
        self.prices = 100 * np.exp(np.concatenate([[0], np.cumsum(returns)]))
        self.num_paths = 10
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-20"
        self.granularity = "D"
        self.model = BootstrapMonteCarlo(
            self.prices,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )

    def test_initialization(self):
        self.assertAlmostEqual(self.model.S0, self.prices[-1])
        self.assertEqual(self.model.num_paths, self.num_paths)
        self.assertEqual(len(self.model.log_returns), len(self.prices) - 1)

    def test_custom_s0(self):
        model = BootstrapMonteCarlo(self.prices, 5, self.start_date, self.end_date, self.granularity, S0=200.0)
        self.assertEqual(model.S0, 200.0)

    def test_simulation_shape(self):
        paths = self.model.simulate()
        self.assertEqual(paths.shape, (self.num_paths, self.model.num_steps))

    def test_initial_values(self):
        paths = self.model.simulate(seed=42)
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.model.S0))

    def test_seed_reproducibility(self):
        a = self.model.simulate(seed=42)
        b = self.model.simulate(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = self.model.simulate(seed=42)
        b = self.model.simulate(seed=99)
        self.assertFalse(np.array_equal(a, b))

    def test_positive_prices(self):
        paths = self.model.simulate(seed=42)
        self.assertTrue(np.all(paths > 0))

    def test_block_bootstrap(self):
        model = BootstrapMonteCarlo(
            self.prices,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
            block_size=5,
        )
        paths = model.simulate(seed=42)
        self.assertEqual(paths.shape, (self.num_paths, model.num_steps))
        self.assertTrue(np.all(paths > 0))

    def test_block_size_property(self):
        self.assertEqual(self.model.block_size, 1)
        self.model.block_size = 10
        self.assertEqual(self.model.block_size, 10)

    def test_milstein_raises(self):
        with self.assertRaises(ValueError):
            self.model.simulate(method="milstein")

    def test_mu_sigma_from_data(self):
        expected_mu = float(np.mean(self.model.log_returns))
        expected_sigma = float(np.std(self.model.log_returns))
        self.assertAlmostEqual(self.model.mu, expected_mu)
        self.assertAlmostEqual(self.model.sigma, expected_sigma)

    def test_granularity_change(self):
        self.model.granularity = "h"
        self.assertEqual(self.model.granularity, "h")
        self.assertEqual(self.model.num_steps, len(self.model.t))


if __name__ == "__main__":
    unittest.main()
