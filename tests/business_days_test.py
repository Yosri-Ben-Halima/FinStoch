import unittest

import numpy as np
import pandas as pd

from FinStoch.processes import GeometricBrownianMotion, OrnsteinUhlenbeck, HestonModel


class TestBusinessDays(unittest.TestCase):
    """Test the business_days flag across process classes."""

    def setUp(self):
        self.S0 = 100.0
        self.mu = 0.05
        self.sigma = 0.2
        self.num_paths = 3
        # 2023-01-01 is a Sunday, 2023-01-15 is a Sunday
        # Calendar days: 15 days (Jan 1-15)
        # Business days: 10 days (Jan 2-13, Mon-Fri)
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-15"
        self.granularity = "D"

    def test_default_uses_calendar_days(self):
        gbm = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )
        self.assertFalse(gbm.business_days)
        # Calendar days: Jan 1 through Jan 15 inclusive = 15 days
        self.assertEqual(gbm.num_steps, 15)

    def test_business_days_excludes_weekends(self):
        gbm = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
            business_days=True,
        )
        self.assertTrue(gbm.business_days)
        # Business days only (no weekends)
        expected = len(pd.bdate_range(self.start_date, self.end_date))
        self.assertEqual(gbm.num_steps, expected)
        self.assertLess(gbm.num_steps, 15)

    def test_business_days_all_weekdays(self):
        gbm = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
            business_days=True,
        )
        for date in gbm.t:
            self.assertIn(date.dayofweek, range(5), f"{date} is a weekend day")

    def test_simulation_shape_with_business_days(self):
        gbm = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
            business_days=True,
        )
        paths = gbm.simulate()
        self.assertEqual(paths.shape, (self.num_paths, gbm.num_steps))
        np.testing.assert_array_equal(paths[:, 0], np.full(self.num_paths, self.S0))

    def test_setter_toggles_business_days(self):
        gbm = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
        )
        calendar_steps = gbm.num_steps

        gbm.business_days = True
        business_steps = gbm.num_steps
        self.assertLess(business_steps, calendar_steps)

        gbm.business_days = False
        self.assertEqual(gbm.num_steps, calendar_steps)

    def test_business_days_ignored_for_non_daily_granularity(self):
        gbm_hourly = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            "2023-01-02",
            "2023-01-02 08:00:00",
            "H",
            business_days=True,
        )
        gbm_hourly_no_bd = GeometricBrownianMotion(
            self.S0,
            self.mu,
            self.sigma,
            self.num_paths,
            "2023-01-02",
            "2023-01-02 08:00:00",
            "H",
            business_days=False,
        )
        # business_days should have no effect on hourly granularity
        self.assertEqual(gbm_hourly.num_steps, gbm_hourly_no_bd.num_steps)

    def test_business_days_works_on_other_processes(self):
        ou = OrnsteinUhlenbeck(
            self.S0,
            self.mu,
            self.sigma,
            0.5,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
            business_days=True,
        )
        expected = len(pd.bdate_range(self.start_date, self.end_date))
        self.assertEqual(ou.num_steps, expected)

        paths = ou.simulate()
        self.assertEqual(paths.shape, (self.num_paths, ou.num_steps))

    def test_business_days_works_on_heston(self):
        heston = HestonModel(
            self.S0,
            0.02,
            self.mu,
            0.3,
            0.04,
            1.5,
            -0.7,
            self.num_paths,
            self.start_date,
            self.end_date,
            self.granularity,
            business_days=True,
        )
        expected = len(pd.bdate_range(self.start_date, self.end_date))
        self.assertEqual(heston.num_steps, expected)

        S, v = heston.simulate()
        self.assertEqual(S.shape, (self.num_paths, heston.num_steps))
        self.assertEqual(v.shape, (self.num_paths, heston.num_steps))


if __name__ == "__main__":
    unittest.main()
