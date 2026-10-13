import unittest

import numpy as np
import pandas as pd

from src.refresh_treasury_5y10y import retain_observed_history
from src.replicate_benchmark_eq6_g10_5y10y import compute_raw_panel


class TreasuryRefreshTests(unittest.TestCase):
    def test_observed_history_retained_but_missing_dates_not_filled(self):
        new = pd.DataFrame({"date": pd.to_datetime(["2023-01-03", "2023-01-05"]),
                            "variable": "rate", "ticker": "TICKER", "value": [1., 3.]})
        old = pd.DataFrame({"date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
                            "variable": "rate", "ticker": "TICKER", "value": [0.9, 2.],
                            "workbook": "baseline"})
        combined, retained = retain_observed_history(new, old)
        self.assertEqual(combined.value.tolist(), [1., 2., 3.])
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained.source.iloc[0], "previous_exact_ticker_quote")
        self.assertNotIn(pd.Timestamp("2023-01-06"), set(combined.date))

    def test_duplicate_new_keys_rejected(self):
        new = pd.DataFrame({"date": pd.to_datetime(["2023-01-03"] * 2),
                            "variable": "rate", "ticker": "TICKER", "value": [1., 2.]})
        old = new.iloc[:1].assign(workbook="baseline")
        with self.assertRaises(pd.errors.MergeError):
            retain_observed_history(new, old)

    def test_scoped_refresh_keeps_same_formula_and_missing_days(self):
        wide = pd.DataFrame({"date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
                             "eur_sofr_bs_5y_basis": [-20., np.nan],
                             "eur_sofr_irs_5y_swap": [3., 3.],
                             "eur_sofr_govt_5y_yield": [2., 2.],
                             "usd_sofr_irs_5y_swap": [4., 4.],
                             "usd_sofr_govt_5y_yield": [3.5, 3.5],
                             "eur_sofr_bs_10y_basis": [-10., -10.],
                             "eur_sofr_irs_10y_swap": [3., 3.],
                             "eur_sofr_govt_10y_yield": [2., 2.],
                             "usd_sofr_irs_10y_swap": [4., 4.],
                             "usd_sofr_govt_10y_yield": [3.5, 3.5]})
        panel = compute_raw_panel(wide, currencies=["EUR"], regimes=["sofr"])
        five = panel[panel.tenor.eq("5y")]
        self.assertEqual(len(five), 1)
        self.assertAlmostEqual(five.rho_raw.iloc[0], -1.2)
        self.assertAlmostEqual(five.x_rf_raw_bps.iloc[0], 20.)
        self.assertAlmostEqual(five.x_govt_raw_bps.iloc[0], -30.)
        self.assertLess(panel.eq6_residual_bps.abs().max(), 1e-8)


if __name__ == "__main__":
    unittest.main()
