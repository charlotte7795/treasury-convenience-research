import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analyze_post_meeting_research import COHORT, OUTCOMES, weekly_hac, weekly_panel


class PostMeetingTests(unittest.TestCase):
    def make_inputs(self):
        rows = []
        dates = pd.bdate_range("2022-11-01", "2023-02-08")
        for i, date in enumerate(dates):
            for currency in COHORT:
                for tenor in ["5y", "10y"]:
                    rows.append({"date": date, "currency": currency, "tenor": tenor,
                                 "regime": "sofr", "is_selected_regime_by_public_break": True,
                                 "x_govt_raw_bps": i + 1., "x_rf_raw_bps": i,
                                 "us_swap_spread_bps": 3., "foreign_swap_spread_bps": 2.})
        return pd.DataFrame(rows), pd.DataFrame({"date": dates, "value": np.arange(len(dates)) + 20.})

    def test_no_partial_last_week_and_additive_means(self):
        raw, vix = self.make_inputs()
        weekly = weekly_panel(raw, vix)
        self.assertEqual(weekly.week.max(), pd.Timestamp("2023-02-03"))
        good = weekly.dropna(subset=OUTCOMES)
        np.testing.assert_allclose(good.x_govt_raw_bps, good.x_rf_raw_bps + good.us_swap_spread_bps - good.foreign_swap_spread_bps)

    def test_missing_week_not_bridged_and_composition_fixed(self):
        raw, vix = self.make_inputs()
        raw = raw[~(raw.currency.eq("NZD") & raw.date.between("2023-01-09", "2023-01-13"))]
        weekly = weekly_panel(raw, vix).set_index(["tenor", "week"])
        self.assertFalse(weekly.loc[("5y", "2023-01-13"), "valid_change"])
        self.assertFalse(weekly.loc[("5y", "2023-01-20"), "valid_change"])
        self.assertTrue(weekly.loc[("5y", "2023-01-27"), "valid_change"])

    def test_benchmark_transition_not_used_as_weekly_change(self):
        raw, vix = self.make_inputs()
        raw.loc[raw.currency.eq("CAD") & raw.date.lt("2023-01-11"), "regime"] = "ibor"
        weekly = weekly_panel(raw, vix).set_index(["tenor", "week"])
        self.assertFalse(weekly.loc[("5y", "2023-01-13"), "valid_change"])
        self.assertFalse(weekly.loc[("5y", "2023-01-20"), "valid_change"])

    def test_weekly_hac_matches_standard_without_gaps(self):
        rng = np.random.default_rng(5)
        x = sm.add_constant(rng.normal(size=90))
        model = sm.OLS(rng.normal(size=90), x).fit()
        dates = pd.date_range("2020-01-03", periods=90, freq="W-FRI")
        expected = sm.stats.sandwich_covariance.cov_hac(model, nlags=26, use_correction=True)
        np.testing.assert_allclose(weekly_hac(model, dates), expected, atol=1e-12)
        gap_dates = dates + pd.to_timedelta(np.where(np.arange(90) >= 40, 7, 0), unit="D")
        self.assertFalse(np.allclose(weekly_hac(model, gap_dates), expected, rtol=1e-8, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
