import unittest

import numpy as np
import pandas as pd

from src.analyze_foreign_swap_spread_rebound import METRICS, contrast, country_monthly


class ForeignSwapSpreadTests(unittest.TestCase):
    def sample(self):
        rows = []
        for start, swap, govt in [("2023-01-02", 2., 1.), ("2026-01-02", 2.5, 2.)]:
            for date in pd.bdate_range(start, periods=15):
                spread = 100 * (swap - govt)
                rows.append({"date": date, "currency": "EUR", "tenor": "5y", "regime": "sofr",
                             "foreign_swap": swap, "foreign_govt_yield": govt,
                             "usd_swap": 1., "usd_govt_yield": .9,
                             "foreign_swap_spread_bps": spread, "us_swap_spread_bps": 10.,
                             "x_rf_raw_bps": 20., "x_govt_raw_bps": 30. - spread})
        return pd.DataFrame(rows)

    def test_same_sample_units_and_additive_bridge(self):
        delta = contrast(country_monthly(self.sample()), "2023-01-31", "2023-12-31").iloc[0]
        self.assertAlmostEqual(delta.foreign_swap_bps, 50.)
        self.assertAlmostEqual(delta.foreign_govt_yield_bps, 100.)
        self.assertAlmostEqual(delta.foreign_swap_spread_bps, -50.)
        self.assertAlmostEqual(delta.x_govt_raw_bps, 50.)
        self.assertAlmostEqual(delta.leg_bridge_error_bps, 0.)
        self.assertAlmostEqual(delta.eq6_bridge_error_bps, 0.)

    def test_short_or_mixed_benchmark_month_is_unavailable_not_zero(self):
        raw = self.sample()
        short = country_monthly(raw.drop(index=0))
        self.assertTrue(short.loc[0, METRICS].isna().all())
        raw.loc[0, "regime"] = "ibor"
        mixed = country_monthly(raw)
        self.assertTrue(mixed.loc[0, METRICS].isna().all())

    def test_duplicate_or_missing_leg_is_rejected(self):
        raw = self.sample()
        with self.assertRaises(AssertionError):
            country_monthly(pd.concat([raw, raw.iloc[[0]]]))
        raw.loc[0, "foreign_swap"] = np.nan
        with self.assertRaises(AssertionError):
            country_monthly(raw)


if __name__ == "__main__":
    unittest.main()
