import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

from src.analyze_treasury_interim import calendar_hac, same_interval_regime
from src.replicate_benchmark_eq6_g10_5y10y import choose_tenor_basis


class TreasuryInterimTests(unittest.TestCase):
    def test_jpy_tona_does_not_use_legacy_libor_basis(self):
        columns = ["jpy_ibor_tbs_5y_jybc5"]
        self.assertEqual(choose_tenor_basis(columns, "JPY", "sofr", "5y"), (None, 0))
        self.assertEqual(choose_tenor_basis(columns, "JPY", "ibor", "5y"), (columns[0], -1))

    def test_change_window_excludes_interior_break_and_missing_month(self):
        labels = pd.Series(["A", "A", "B", "A", "A", None, "A", "A", "A"])
        expected = [False, False, False, False, False, False, False, False, True]
        self.assertEqual(same_interval_regime(labels, 2).tolist(), expected)

    def test_calendar_hac_matches_statsmodels_without_gaps(self):
        rng = np.random.default_rng(83)
        x = sm.add_constant(pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"]))
        model = sm.OLS(rng.normal(size=80), x).fit()
        dates = pd.date_range("2010-01-31", periods=80, freq="ME")
        np.testing.assert_allclose(calendar_hac(model, dates, 12), cov_hac(model, nlags=12), atol=1e-12)

    def test_calendar_hac_does_not_join_observations_across_gaps(self):
        x = sm.add_constant(np.array([1., 4., 2., 5., 3.]))
        model = sm.OLS(np.array([3., 1., 4., 8., 2.]), x).fit()
        dates = pd.DatetimeIndex(["2000-01-31", "2000-03-31", "2000-05-31", "2000-07-31", "2000-09-30"])
        scores = x * model.resid[:, None]
        expected = model.normalized_cov_params @ (scores.T @ scores) @ model.normalized_cov_params * 5 / 3
        np.testing.assert_allclose(calendar_hac(model, dates, 1), expected, atol=1e-12)
        self.assertFalse(np.allclose(calendar_hac(model, dates, 1), cov_hac(model, nlags=1)))


if __name__ == "__main__":
    unittest.main()
