import unittest
import numpy as np
from FinStoch.processes import OrnsteinUhlenbeck


class TestOrnsteinUhlenbeck(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        self.theta = 0.5
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.ou = OrnsteinUhlenbeck(
            self.S0,
            self.mu,
            self.sigma,
            self.theta,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )

    def test_initialization(self):
        # Assertions to verify the initialization values
        self.assertEqual(self.ou.S0, self.S0)
        self.assertEqual(self.ou.mu, self.mu)
        self.assertEqual(self.ou.sigma, self.sigma)
        self.assertEqual(self.ou.theta, self.theta)
        self.assertEqual(self.ou.num_paths, self.num_paths)
        self.assertEqual(self.ou.start_date, self.start_date)
        self.assertEqual(self.ou.end_date, self.end_date)
        self.assertEqual(self.ou.granularity, self.granularity)

        # Check if the duration and number of steps are calculated correctly
        self.assertAlmostEqual(self.ou.T, 0.024640657084188913)
        self.assertEqual(self.ou.num_steps, 10)

    def test_simulation(self):
        # Simulate paths of the Ornstein Uhlenbeck model
        paths = self.ou.simulate()

        # Check shape of the simulation output
        self.assertEqual(paths.shape, (self.num_paths, self.ou.num_steps))

        # Check the first value in each path equals the initial asset value
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_property_setters(self):
        # Test changing S0
        new_S0 = 150.0
        self.ou.S0 = new_S0
        self.assertEqual(self.ou.S0, new_S0)

        # Test changing mu
        new_mu = 0.1
        self.ou.mu = new_mu
        self.assertEqual(self.ou.mu, new_mu)

        # Test changing sigma
        new_sigma = 0.3
        self.ou.sigma = new_sigma
        self.assertEqual(self.ou.sigma, new_sigma)

        # Test changing theta
        new_theta = 0.6
        self.ou.theta = new_theta
        self.assertEqual(self.ou.theta, new_theta)

        # Test updating start_date recalculates time-related properties
        new_start_date = "2023-01-05"
        self.ou.start_date = new_start_date
        self.assertEqual(self.ou.start_date, new_start_date)
        self.assertEqual(self.ou.num_steps, len(self.ou.t))

        # Test updating end_date recalculates time-related properties
        new_end_date = "2023-01-15"
        self.ou.end_date = new_end_date
        self.assertEqual(self.ou.end_date, new_end_date)
        self.assertEqual(self.ou.num_steps, len(self.ou.t))

    def test_granularity_change(self):
        # Test updating granularity recalculates time steps and related properties
        new_granularity = "H"
        self.ou.granularity = new_granularity
        self.assertEqual(self.ou.granularity, new_granularity)
        self.assertEqual(self.ou.num_steps, len(self.ou.t))


if __name__ == "__main__":
    unittest.main()
