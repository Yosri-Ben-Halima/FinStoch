import unittest
import numpy as np
from processes.gbm import GeometricBrownianMotion

class TestGeometricBrownianMotion(unittest.TestCase):

    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        self.T = 1.0
        self.num_steps = 100
        self.num_paths = 10
        self.gbm = GeometricBrownianMotion(self.S0, self.mu, self.sigma, self.T, self.num_steps, self.num_paths)

    def test_initial_parameters(self):
        self.assertEqual(self.gbm.S0, self.S0)
        self.assertEqual(self.gbm.mu, self.mu)
        self.assertEqual(self.gbm.sigma, self.sigma)
        self.assertEqual(self.gbm.T, self.T)
        self.assertEqual(self.gbm.num_steps, self.num_steps)
        self.assertEqual(self.gbm.num_paths, self.num_paths)

    def test_simulate(self):
        simulated_paths = self.gbm.simulate()
        self.assertEqual(simulated_paths.shape, (self.num_paths, self.num_steps))
        self.assertTrue(np.all(simulated_paths[:, 0] == self.S0))

    def test_setters(self):
        new_S0 = 120.0
        new_mu = 0.1
        new_sigma = 0.3
        new_T = 2.0
        new_num_steps = 200
        new_num_paths = 20

        self.gbm.S0 = new_S0
        self.gbm.mu = new_mu
        self.gbm.sigma = new_sigma
        self.gbm.T = new_T
        self.gbm.num_steps = new_num_steps
        self.gbm.num_paths = new_num_paths

        self.assertEqual(self.gbm.S0, new_S0)
        self.assertEqual(self.gbm.mu, new_mu)
        self.assertEqual(self.gbm.sigma, new_sigma)
        self.assertEqual(self.gbm.T, new_T)
        self.assertEqual(self.gbm.num_steps, new_num_steps)
        self.assertEqual(self.gbm.num_paths, new_num_paths)

    def test_dt_and_t(self):
        self.assertAlmostEqual(self.gbm.dt, self.T / self.num_steps)
        self.assertTrue(np.allclose(self.gbm.t, np.linspace(0, self.T, self.num_steps)))

        new_T = 2.0
        new_num_steps = 200
        self.gbm.T = new_T
        self.gbm.num_steps = new_num_steps

        self.assertAlmostEqual(self.gbm.dt, new_T / new_num_steps)
        self.assertTrue(np.allclose(self.gbm.t, np.linspace(0, new_T, new_num_steps)))

if __name__ == '__main__':
    unittest.main()