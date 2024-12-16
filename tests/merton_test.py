import unittest
import numpy as np
from FinStoch.processes import MertonJumpDiffusion


class TestMertonModel(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        lambda_j = 1
        mu_j = 0.02
        sigma_j = 0.1
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.merton = MertonJumpDiffusion(
            self.S0,
            self.mu,
            self.sigma,
            lambda_j,
            mu_j,
            sigma_j,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )

    def test_initialization(self):
        # Assertions to verify the initialization values
        self.assertEqual(self.merton.S0, self.S0)
        self.assertEqual(self.merton.mu, self.mu)
        self.assertEqual(self.merton.sigma, self.sigma)
        self.assertEqual(self.merton.num_paths, self.num_paths)
        self.assertEqual(self.merton.start_date, self.start_date)
        self.assertEqual(self.merton.end_date, self.end_date)
        self.assertEqual(self.merton.granularity, self.granularity)

        # Check if the duration and number of steps are calculated correctly
        self.assertAlmostEqual(self.merton.T, 0.024640657084188913)
        self.assertEqual(self.merton.num_steps, 10)

    def test_simulation(self):
        # Simulate paths of the geometric Brownian motion
        paths = self.merton.simulate()

        # Check shape of the simulation output
        self.assertEqual(paths.shape, (self.num_paths, self.merton.num_steps))

        # Check the first value in each path equals the initial asset value
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_property_setters(self):
        # Test changing S0
        new_S0 = 150.0
        self.merton.S0 = new_S0
        self.assertEqual(self.merton.S0, new_S0)

        # Test changing mu
        new_mu = 0.1
        self.merton.mu = new_mu
        self.assertEqual(self.merton.mu, new_mu)

        # Test changing sigma
        new_sigma = 0.3
        self.merton.sigma = new_sigma
        self.assertEqual(self.merton.sigma, new_sigma)

        # Test updating start_date recalculates time-related properties
        new_start_date = "2023-01-05"
        self.merton.start_date = new_start_date
        self.assertEqual(self.merton.start_date, new_start_date)
        self.assertEqual(self.merton.num_steps, len(self.merton.t))

        # Test changing end_date recalculates time-related properties
        new_end_date = "2023-01-15"
        self.merton.end_date = new_end_date
        self.assertEqual(self.merton.end_date, new_end_date)
        self.assertEqual(self.merton.num_steps, len(self.merton.t))

    def test_granularity_change(self):
        # Test updating granularity recalculates time steps and related properties
        new_granularity = "H"
        self.merton.granularity = new_granularity
        self.assertEqual(self.merton.granularity, new_granularity)
        self.assertEqual(self.merton.num_steps, len(self.merton.t))


if __name__ == "__main__":
    unittest.main()
