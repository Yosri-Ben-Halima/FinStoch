import unittest
import numpy as np
from FinStoch.processes import ConstantElasricityOfVariance


class TestConstantElasricityOfVariance(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        self.gamma = 0.5
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.cev = ConstantElasricityOfVariance(
            self.S0,
            self.mu,
            self.sigma,
            self.gamma,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )

    def test_initialization(self):
        # Assertions to verify the initialization values
        self.assertEqual(self.cev.S0, self.S0)
        self.assertEqual(self.cev.mu, self.mu)
        self.assertEqual(self.cev.sigma, self.sigma)
        self.assertEqual(self.cev.gamma, self.gamma)
        self.assertEqual(self.cev.num_paths, self.num_paths)
        self.assertEqual(self.cev.start_date, self.start_date)
        self.assertEqual(self.cev.end_date, self.end_date)
        self.assertEqual(self.cev.granularity, self.granularity)

        # Check if the duration and number of steps are calculated correctly
        self.assertAlmostEqual(self.cev.T, 0.024640657084188913)
        self.assertEqual(self.cev.num_steps, 10)

    def test_simulation(self):
        # Simulate paths of the constant elasricity of variance model
        paths = self.cev.simulate()

        # Check shape of the simulation output
        self.assertEqual(paths.shape, (self.num_paths, self.cev.num_steps))

        # Check the first value in each path equals the initial asset value
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_property_setters(self):
        # Test changing S0
        new_S0 = 150.0
        self.cev.S0 = new_S0
        self.assertEqual(self.cev.S0, new_S0)

        # Test changing mu
        new_mu = 0.1
        self.cev.mu = new_mu
        self.assertEqual(self.cev.mu, new_mu)

        # Test changing sigma
        new_sigma = 0.3
        self.cev.sigma = new_sigma
        self.assertEqual(self.cev.sigma, new_sigma)

        # Test changing gamma
        new_gamma = 0.7
        self.cev.gamma = new_gamma
        self.assertEqual(self.cev.gamma, new_gamma)

        # Test updating start_date recalculates time-related properties
        new_start_date = "2023-01-05"
        self.cev.start_date = new_start_date
        self.assertEqual(self.cev.start_date, new_start_date)
        self.assertEqual(self.cev.num_steps, len(self.cev.t))

        # Test changing end_date recalculates time-related properties
        new_end_date = "2023-01-15"
        self.cev.end_date = new_end_date
        self.assertEqual(self.cev.end_date, new_end_date)
        self.assertEqual(self.cev.num_steps, len(self.cev.t))

    def test_granularity_change(self):
        # Test updating granularity recalculates time steps and related properties
        new_granularity = "H"
        self.cev.granularity = new_granularity
        self.assertEqual(self.cev.granularity, new_granularity)
        self.assertEqual(self.cev.num_steps, len(self.cev.t))


if __name__ == "__main__":
    unittest.main()
