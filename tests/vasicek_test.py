import unittest
import numpy as np
from FinStoch.processes import VasicekModel


class TestVasicekModel(unittest.TestCase):
    def setUp(self):
        self.S0 = 0.05
        self.mu = 0.03
        self.sigma = 0.01
        self.a = 0.5
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.vasicek = VasicekModel(
            S0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            a=self.a,
            num_paths=self.num_paths,
            start_date=self.start_date,
            end_date=self.end_date,
            granularity=self.granularity,
        )

    def test_initialization(self):
        self.assertEqual(self.vasicek.S0, self.S0)
        self.assertEqual(self.vasicek.mu, self.mu)
        self.assertEqual(self.vasicek.sigma, self.sigma)
        self.assertEqual(self.vasicek.a, self.a)
        self.assertEqual(self.vasicek.num_paths, self.num_paths)
        self.assertEqual(self.vasicek.start_date, self.start_date)
        self.assertEqual(self.vasicek.end_date, self.end_date)
        self.assertEqual(self.vasicek.granularity, self.granularity)
        self.assertAlmostEqual(self.vasicek.T, 0.024640657084188913)
        self.assertEqual(self.vasicek.num_steps, 10)

    def test_simulation(self):
        paths = self.vasicek.simulate()
        self.assertEqual(paths.shape, (self.num_paths, self.vasicek.num_steps))
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_property_setters(self):
        new_a = 1.0
        self.vasicek.a = new_a
        self.assertEqual(self.vasicek.a, new_a)

        new_S0 = 0.04
        self.vasicek.S0 = new_S0
        self.assertEqual(self.vasicek.S0, new_S0)

        new_start_date = "2023-01-05"
        self.vasicek.start_date = new_start_date
        self.assertEqual(self.vasicek.start_date, new_start_date)
        self.assertEqual(self.vasicek.num_steps, len(self.vasicek.t))

    def test_granularity_change(self):
        new_granularity = "h"
        self.vasicek.granularity = new_granularity
        self.assertEqual(self.vasicek.granularity, new_granularity)
        self.assertEqual(self.vasicek.num_steps, len(self.vasicek.t))


if __name__ == "__main__":
    unittest.main()
