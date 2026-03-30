import unittest
import numpy as np
from FinStoch.processes import VarianceGammaProcess


class TestVarianceGammaProcess(unittest.TestCase):
    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        self.theta = -0.1
        self.nu = 0.2
        self.num_paths = 5
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-10"
        self.granularity = "D"
        self.vg = VarianceGammaProcess(
            S0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            theta=self.theta,
            nu=self.nu,
            num_paths=self.num_paths,
            start_date=self.start_date,
            end_date=self.end_date,
            granularity=self.granularity,
        )

    def test_initialization(self):
        self.assertEqual(self.vg.S0, self.S0)
        self.assertEqual(self.vg.mu, self.mu)
        self.assertEqual(self.vg.sigma, self.sigma)
        self.assertEqual(self.vg.theta, self.theta)
        self.assertEqual(self.vg.nu, self.nu)
        self.assertEqual(self.vg.num_paths, self.num_paths)
        self.assertAlmostEqual(self.vg.T, 0.024640657084188913)
        self.assertEqual(self.vg.num_steps, 10)

    def test_simulation(self):
        paths = self.vg.simulate()
        self.assertEqual(paths.shape, (self.num_paths, self.vg.num_steps))
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_property_setters(self):
        self.vg.theta = -0.2
        self.assertEqual(self.vg.theta, -0.2)

        self.vg.nu = 0.3
        self.assertEqual(self.vg.nu, 0.3)

        self.vg.start_date = "2023-01-05"
        self.assertEqual(self.vg.start_date, "2023-01-05")
        self.assertEqual(self.vg.num_steps, len(self.vg.t))

    def test_granularity_change(self):
        self.vg.granularity = "h"
        self.assertEqual(self.vg.granularity, "h")
        self.assertEqual(self.vg.num_steps, len(self.vg.t))

    def test_milstein_raises(self):
        with self.assertRaises(ValueError):
            self.vg.simulate(method="milstein")


if __name__ == "__main__":
    unittest.main()
