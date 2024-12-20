import unittest
import numpy as np
from FinStoch.processes import GeometricBrownianMotion


class TestGeometricBrownianMotion(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.gbm = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )

    def test_initialization(self):
        # Assertions to verify the initialization values
        self.assertEqual(self.gbm.S0, self.S0)
        self.assertEqual(self.gbm.mu, self.mu)
        self.assertEqual(self.gbm.sigma, self.sigma)
        self.assertEqual(self.gbm.num_paths, self.num_paths)
        self.assertEqual(self.gbm.start_date, self.start_date)
        self.assertEqual(self.gbm.end_date, self.end_date)
        self.assertEqual(self.gbm.granularity, self.granularity)

        # Check if the duration and number of steps are calculated correctly
        self.assertAlmostEqual(self.gbm.T, 0.024640657084188913)
        self.assertEqual(self.gbm.num_steps, 10)

    def test_simulation(self):
        # Simulate paths of the geometric Brownian motion
        paths = self.gbm.simulate()

        # Check shape of the simulation output
        self.assertEqual(paths.shape, (self.num_paths, self.gbm.num_steps))

        # Check the first value in each path equals the initial asset value
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_property_setters(self):
        # Test changing S0
        new_S0 = 150.0
        self.gbm.S0 = new_S0
        self.assertEqual(self.gbm.S0, new_S0)

        # Test changing mu
        new_mu = 0.1
        self.gbm.mu = new_mu
        self.assertEqual(self.gbm.mu, new_mu)

        # Test changing sigma
        new_sigma = 0.3
        self.gbm.sigma = new_sigma
        self.assertEqual(self.gbm.sigma, new_sigma)

        # Test updating start_date recalculates time-related properties
        new_start_date = "2023-01-05"
        self.gbm.start_date = new_start_date
        self.assertEqual(self.gbm.start_date, new_start_date)
        self.assertEqual(self.gbm.num_steps, len(self.gbm.t))

        # Test changing end_date recalculates time-related properties
        new_end_date = "2023-01-15"
        self.gbm.end_date = new_end_date
        self.assertEqual(self.gbm.end_date, new_end_date)
        self.assertEqual(self.gbm.num_steps, len(self.gbm.t))

    def test_granularity_change(self):
        # Test updating granularity recalculates time steps and related properties
        new_granularity = "H"
        self.gbm.granularity = new_granularity
        self.assertEqual(self.gbm.granularity, new_granularity)
        self.assertEqual(self.gbm.num_steps, len(self.gbm.t))


if __name__ == "__main__":
    unittest.main()
