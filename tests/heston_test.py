import unittest
import numpy as np
from FinStoch.processes import HestonModel


class TestHestonModel(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.v0 = 0.2
        self.mu = 0.05
        self.sigma = 0.2
        self.theta = 0.5
        self.kappa = 0.7
        self.rho = 0.6
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.heston = HestonModel(
            self.S0,
            self.v0,
            self.mu,
            self.sigma,
            self.theta,
            self.kappa,
            self.rho,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )

    def test_initialization(self):
        # Assertions to verify the initialization values
        self.assertEqual(self.heston.S0, self.S0)
        self.assertEqual(self.heston.v0, self.v0)
        self.assertEqual(self.heston.mu, self.mu)
        self.assertEqual(self.heston.sigma, self.sigma)
        self.assertEqual(self.heston.theta, self.theta)
        self.assertEqual(self.heston.kappa, self.kappa)
        self.assertEqual(self.heston.rho, self.rho)
        self.assertEqual(self.heston.num_paths, self.num_paths)
        self.assertEqual(self.heston.start_date, self.start_date)
        self.assertEqual(self.heston.end_date, self.end_date)
        self.assertEqual(self.heston.granularity, self.granularity)

        # Check if the duration and number of steps are calculated correctly
        self.assertAlmostEqual(self.heston.T, 0.024640657084188913)
        self.assertEqual(self.heston.num_steps, 10)

    def test_simulation(self):
        # Simulate paths of the geometric Brownian motion
        S_paths, v_paths = self.heston.simulate()

        # Check shape of the simulation output
        self.assertEqual(S_paths.shape, (self.num_paths, self.heston.num_steps))

        # Check the first value in each path equals the initial asset value
        np.testing.assert_array_equal(S_paths[:, 0], np.full(self.num_paths, self.S0))

        # Check shape of the simulation output
        self.assertEqual(v_paths.shape, (self.num_paths, self.heston.num_steps))

        # Check the first value in each path equals the initial asset value
        np.testing.assert_array_equal(v_paths[:, 0], np.full(self.num_paths, self.v0))

    def test_property_setters(self):
        # Test changing S0
        new_S0 = 150.0
        self.heston.S0 = new_S0
        self.assertEqual(self.heston.S0, new_S0)

        # Test changing v0
        new_v0 = 0.3
        self.heston.v0 = new_v0
        self.assertEqual(self.heston.v0, new_v0)

        # Test changing mu
        new_mu = 0.06
        self.heston.mu = new_mu
        self.assertEqual(self.heston.mu, new_mu)

        # Test changing sigma
        new_sigma = 0.35
        self.heston.sigma = new_sigma
        self.assertEqual(self.heston.sigma, new_sigma)

        # Test changing theta
        new_theta = 0.4
        self.heston.theta = new_theta
        self.assertEqual(self.heston.theta, new_theta)

        # Test changing kappa
        new_kappa = 0.65
        self.heston.kappa = new_kappa
        self.assertEqual(self.heston.kappa, new_kappa)

        # Test changing rho
        new_rho = 0.5
        self.heston.rho = new_rho
        self.assertEqual(self.heston.rho, new_rho)

        # Test changing start_date
        new_start_date = "2023-01-05"
        self.heston.start_date = new_start_date
        self.assertEqual(self.heston.start_date, new_start_date)

        # Test changing end_date
        new_end_date = "2023-01-15"
        self.heston.end_date = new_end_date
        self.assertEqual(self.heston.end_date, new_end_date)

        # Test changing num_paths
        new_num_paths = 10
        self.heston.num_paths = new_num_paths
        self.assertEqual(self.heston.num_paths, new_num_paths)

    def test_granularity_change(self):
        # Test updating granularity recalculates time steps and related properties
        new_granularity = "H"
        self.heston.granularity = new_granularity
        self.assertEqual(self.heston.granularity, new_granularity)
        self.assertEqual(self.heston.num_steps, len(self.heston.t))


if __name__ == "__main__":
    unittest.main()
