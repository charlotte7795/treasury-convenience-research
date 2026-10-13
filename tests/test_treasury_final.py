import unittest

import numpy as np
import pandas as pd

from src.analyze_treasury_final import period_changes
from src.analyze_treasury_interim import COHORT


class TreasuryFinalTests(unittest.TestCase):
    def test_additive_country_means_and_period_boundaries(self):
        rows = []
        for month, move in [("2023-01-31", 0.), ("2026-01-31", 10.), ("2026-10-31", 999.)]:
            for ccy in COHORT:
                for tenor in ["5y", "10y"]:
                    rows.append({"month": pd.Timestamp(month), "currency": ccy, "tenor": tenor,
                                 "foreign_swap_bps": 200.-move, "foreign_govt_yield_bps": 100.,
                                 "usd_swap_bps": 400., "usd_govt_yield_bps": 300.,
                                 "foreign_swap_spread_bps": 100.-move, "us_swap_spread_bps": 100.,
                                 "x_rf_raw_bps": 20., "x_govt_raw_bps": 20.+move,
                                 "diff_y_raw_bps": -200., "rho_raw_bps": -220.-move})
        delta = period_changes(pd.DataFrame(rows), ("2023-01-31", "2023-12-31"),
                               ("2026-01-31", "2026-09-30"))
        self.assertEqual(len(delta), 16)
        np.testing.assert_allclose(delta.x_govt_raw_bps, 10.)
        np.testing.assert_allclose(delta.foreign_swap_spread_bps, -10.)
        np.testing.assert_allclose(delta.eq5_bridge_error_bps, 0.)
        np.testing.assert_allclose(delta.eq6_bridge_error_bps, 0.)


if __name__ == "__main__":
    unittest.main()
