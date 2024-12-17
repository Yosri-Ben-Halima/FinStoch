import unittest
import numpy as np
from FinStoch.processes import CoxIngersollRoss


class TestCoxIngersollRoss(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        self.theta = 0.5
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.cir = CoxIngersollRoss(
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
        self.assertEqual(self.cir.S0, self.S0)
        self.assertEqual(self.cir.mu, self.mu)
        self.assertEqual(self.cir.sigma, self.sigma)
        self.assertEqual(self.cir.theta, self.theta)
        self.assertEqual(self.cir.num_paths, self.num_paths)
        self.assertEqual(self.cir.start_date, self.start_date)
        self.assertEqual(self.cir.end_date, self.end_date)
        self.assertEqual(self.cir.granularity, self.granularity)

        # Check if the duration and number of steps are calculated correctly
        self.assertAlmostEqual(self.cir.T, 0.024640657084188913)
        self.assertEqual(self.cir.num_steps, 10)

    def test_simulation(self):
        # Simulate paths of the CIR Model
        paths = self.cir.simulate()

        # Check shape of the simulation output
        self.assertEqual(paths.shape, (self.num_paths, self.cir.num_steps))

        # Check the first value in each path equals the initial asset value
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_property_setters(self):
        # Test changing S0
        new_S0 = 150.0
        self.cir.S0 = new_S0
        self.assertEqual(self.cir.S0, new_S0)

        # Test changing mu
        new_mu = 0.1
        self.cir.mu = new_mu
        self.assertEqual(self.cir.mu, new_mu)

        # Test changing sigma
        new_sigma = 0.3
        self.cir.sigma = new_sigma
        self.assertEqual(self.cir.sigma, new_sigma)

        # Test changing theta
        new_theta = 0.7
        self.cir.theta = new_theta
        self.assertEqual(self.cir.theta, new_theta)

        # Test updating start_date recalculates time-related properties
        new_start_date = "2023-01-05"
        self.cir.start_date = new_start_date
        self.assertEqual(self.cir.start_date, new_start_date)
        self.assertEqual(self.cir.num_steps, len(self.cir.t))

        # Test changing end_date recalculates time-related properties
        new_end_date = "2023-01-15"
        self.cir.end_date = new_end_date
        self.assertEqual(self.cir.end_date, new_end_date)
        self.assertEqual(self.cir.num_steps, len(self.cir.t))

    def test_granularity_change(self):
        # Test updating granularity recalculates time steps and related properties
        new_granularity = "H"
        self.cir.granularity = new_granularity
        self.assertEqual(self.cir.granularity, new_granularity)
        self.assertEqual(self.cir.num_steps, len(self.cir.t))


if __name__ == "__main__":
    unittest.main()
