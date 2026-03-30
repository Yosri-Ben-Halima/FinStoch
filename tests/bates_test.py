import unittest
import numpy as np
from FinStoch.processes import BatesModel


class TestBatesModel(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.v0 = 0.04
        self.mu = 0.05
        self.sigma = 0.3
        self.theta = 0.04
        self.kappa = 2.0
        self.rho = -0.7
        self.lambda_j = 0.1
        self.mu_j = -0.05
        self.sigma_j = 0.1
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.bates = BatesModel(
            S0=self.S0,
            v0=self.v0,
            mu=self.mu,
            sigma=self.sigma,
            theta=self.theta,
            kappa=self.kappa,
            rho=self.rho,
            lambda_j=self.lambda_j,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
            num_paths=self.num_paths,
            start_date=self.start_date,
            end_date=self.end_date,
            granularity=self.granularity,
        )

    def test_initialization(self):
        self.assertEqual(self.bates.S0, self.S0)
        self.assertEqual(self.bates.v0, self.v0)
        self.assertEqual(self.bates.mu, self.mu)
        self.assertEqual(self.bates.sigma, self.sigma)
        self.assertEqual(self.bates.theta, self.theta)
        self.assertEqual(self.bates.kappa, self.kappa)
        self.assertEqual(self.bates.rho, self.rho)
        self.assertEqual(self.bates.lambda_j, self.lambda_j)
        self.assertEqual(self.bates.mu_j, self.mu_j)
        self.assertEqual(self.bates.sigma_j, self.sigma_j)

    def test_simulation(self):
        result = self.bates.simulate()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        S, v = result
        self.assertEqual(S.shape, (self.num_paths, self.bates.num_steps))
        self.assertEqual(v.shape, (self.num_paths, self.bates.num_steps))
        np.testing.assert_array_equal(S[:, 0], np.full(self.num_paths, self.S0))
        np.testing.assert_array_equal(v[:, 0], np.full(self.num_paths, self.v0))

    def test_property_setters(self):
        self.bates.v0 = 0.05
        self.assertEqual(self.bates.v0, 0.05)

        self.bates.theta = 0.06
        self.assertEqual(self.bates.theta, 0.06)

        self.bates.kappa = 3.0
        self.assertEqual(self.bates.kappa, 3.0)

        self.bates.rho = -0.5
        self.assertEqual(self.bates.rho, -0.5)

        self.bates.lambda_j = 0.2
        self.assertEqual(self.bates.lambda_j, 0.2)

        self.bates.mu_j = -0.1
        self.assertEqual(self.bates.mu_j, -0.1)

        self.bates.sigma_j = 0.15
        self.assertEqual(self.bates.sigma_j, 0.15)

        self.bates.start_date = "2023-01-05"
        self.assertEqual(self.bates.num_steps, len(self.bates.t))

    def test_granularity_change(self):
        self.bates.granularity = "h"
        self.assertEqual(self.bates.granularity, "h")
        self.assertEqual(self.bates.num_steps, len(self.bates.t))


if __name__ == "__main__":
    unittest.main()
