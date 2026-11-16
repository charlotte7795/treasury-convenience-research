import unittest

import numpy as np
import pandas as pd

from src.analyze_credit_risk_extension import FACTORS, fit_models, make_weekly
from src.analyze_post_meeting_research import COHORT, OUTCOMES


class CreditRiskTests(unittest.TestCase):
    def daily_inputs(self):
        dates = pd.bdate_range("2023-01-02", "2023-04-05")
        rows = []
        for i, date in enumerate(dates):
            for currency in COHORT:
                for tenor in ["5y", "10y"]:
                    rows.append({"date": date, "currency": currency, "tenor": tenor,
                                 "regime": "sofr", "is_selected_regime_by_public_break": True,
                                 "x_govt_raw_bps": i + 1., "x_rf_raw_bps": i,
                                 "us_swap_spread_bps": 3., "foreign_swap_spread_bps": 2.})
        market = pd.DataFrame({"vix_index": np.arange(len(dates)) + 20.,
                               "spx_index": np.arange(len(dates)) + 4000.,
                               "ig_oas": np.arange(len(dates)) / 100 + 1.,
                               "hy_oas": np.arange(len(dates)) / 50 + 3.,
                               "move_index": 80., "bbdxy_index": 1000.}, index=dates)
        return pd.DataFrame(rows), market

    def test_week_cutoff_and_bps_conversion(self):
        raw, market = self.daily_inputs()
        weekly = make_weekly(raw, market)
        self.assertEqual(weekly.week.max(), pd.Timestamp("2023-03-31"))
        sample = weekly[weekly.cohort.eq("fixed8") & weekly.tenor.eq("5y") & weekly.valid_change]
        np.testing.assert_allclose(sample.d_ig_bps, 5.)
        np.testing.assert_allclose(sample.d_hy_bps, 10.)
        np.testing.assert_allclose(sample.d_x_govt_raw_bps, sample.d_x_rf_raw_bps + sample.d_us_swap_spread_bps - sample.d_foreign_swap_spread_bps)

    def test_missing_equity_week_not_bridged_or_zero_filled(self):
        raw, market = self.daily_inputs()
        market.loc["2023-02-06":"2023-02-10", "spx_index"] = np.nan
        weekly = make_weekly(raw, market).set_index(["cohort", "tenor", "week"])
        self.assertFalse(weekly.loc[("fixed8", "5y", "2023-02-10"), "valid_change"])
        self.assertFalse(weekly.loc[("fixed8", "5y", "2023-02-17"), "valid_change"])
        self.assertTrue(weekly.loc[("fixed8", "5y", "2023-02-24"), "valid_change"])

    def test_exclusion_cohort_preserves_calendar_sample(self):
        raw, market = self.daily_inputs()
        raw = raw[~(raw.currency.eq("NZD") & raw.date.eq("2023-02-13"))]
        weekly = make_weekly(raw, market)
        a = weekly[weekly.cohort.eq("fixed8")].set_index(["tenor", "week"])
        b = weekly[weekly.cohort.eq("exclude_NZD")].set_index(["tenor", "week"])
        pd.testing.assert_series_equal(a.valid_change, b.valid_change)
        pd.testing.assert_series_equal(a.n_joint_days, b.n_joint_days)

    def test_nested_samples_and_hy_signal(self):
        dates = pd.date_range("2010-01-08", "2026-09-25", freq="W-FRI")
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(len(dates), len(FACTORS))), columns=FACTORS)
        frame["week"] = dates
        frame["valid_change"] = True
        frame["signature"] = np.where(dates < "2022-01-01", "old", "new")
        frame["cohort"], frame["tenor"] = "fixed8", "5y"
        frame["d_x_govt_raw_bps"] = 2 * frame.dlog_vix + 3 * frame.d_hy_bps + rng.normal(size=len(frame))
        frame["d_x_rf_raw_bps"] = frame.dlog_vix
        frame["d_us_swap_spread_bps"] = rng.normal(size=len(frame))
        frame["d_foreign_swap_spread_bps"] = frame.d_x_rf_raw_bps + frame.d_us_swap_spread_bps - frame.d_x_govt_raw_bps
        coefficients, fits, increments = fit_models(frame)
        self.assertTrue(increments.n_weeks_base.eq(increments.n_weeks_hy).all())
        self.assertTrue(increments.delta_r_squared.ge(0).all())
        hy = coefficients[coefficients.window.eq("post2023") & coefficients.spec.eq("plus_HY")
                          & coefficients.outcome.eq("x_govt_raw_bps") & coefficients.factor.eq("d_hy_bps")
                          & coefficients.hac_weeks.eq(26)].iloc[0]
        self.assertGreater(hy.ci95_low, 2.)
        self.assertLess(hy.ci95_high, 4.)


if __name__ == "__main__":
    unittest.main()
