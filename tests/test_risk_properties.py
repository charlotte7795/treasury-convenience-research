import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analyze_credit_risk_extension import FACTORS
from src.analyze_risk_properties import (
    COHORT, OUTCOMES, block_bootstrap, joint_tests, joint_wald, paper_comparison, stress_rule,
)


class RiskPropertiesTests(unittest.TestCase):
    def test_wald_includes_cross_covariance(self):
        rng = np.random.default_rng(22)
        x = sm.add_constant(rng.normal(size=(120, 3)))
        model = sm.OLS(x @ [1., 2., -1., .5] + rng.normal(size=120), x).fit()
        restriction = np.array([[0., 1., 1., 0.], [0., 0., 0., 1.]])
        covariance = model.cov_params()
        statistic, p = joint_wald(model.params, covariance, restriction)
        reference = model.wald_test(restriction, cov_p=covariance, use_f=False, scalar=True)
        self.assertAlmostEqual(statistic, reference.statistic)
        self.assertAlmostEqual(p, reference.pvalue)

    def test_paper_comparison_is_univariate_and_stops_in_june(self):
        dates = pd.date_range("2010-01-08", "2025-07-04", freq="W-FRI")
        rng = np.random.default_rng(8)
        factors = pd.DataFrame(rng.normal(size=(len(dates), len(FACTORS))), index=dates, columns=FACTORS)
        levels = (2 * factors.dlog_vix).cumsum()
        raw = pd.MultiIndex.from_product([dates, COHORT, ["5y", "10y"]], names=["date", "currency", "tenor"]).to_frame(index=False)
        raw["is_selected_regime_by_public_break"] = True
        raw["x_govt_raw_bps"] = raw.date.map(levels)
        public = pd.MultiIndex.from_product([dates, [*COHORT, "JPY", "NOK"], ["5y", "10y"]], names=["date", "currency", "tenor"]).to_frame(index=False)
        public["cip_govt"] = public.date.map(levels)
        weekly = pd.concat([factors.assign(week=dates, cohort="fixed8", tenor=t, valid_change=True,
                                          d_x_govt_raw_bps=2 * factors.dlog_vix) for t in ["5y", "10y"]], ignore_index=True)
        result = paper_comparison(raw, public, pd.DataFrame({"observed": 1.}, index=dates), weekly)
        self.assertEqual(len(result), 54)
        self.assertLessEqual(result.last_week.max(), pd.Timestamp("2025-06-27"))
        np.testing.assert_allclose(result[result.factor.eq("dlog_vix")].raw_beta, 2.)
        self.assertTrue(result.groupby(["tenor", "period"]).n_weeks.nunique().eq(1).all())

    def test_stress_rule_uses_only_historical_vix_and_keeps_gaps(self):
        dates = pd.bdate_range("2010-01-04", "2026-10-05")
        vix = pd.Series(20 * np.exp(.2 * np.sin(np.arange(len(dates)) / 8)), index=dates)
        missing = pd.bdate_range("2022-04-04", "2022-04-08")
        vix = vix.drop(missing)
        flags, receipt = stress_rule(vix)
        changed = vix.copy()
        changed.loc["2023-01-01":] *= np.exp(np.sin(np.arange(len(changed.loc["2023-01-01":]))))
        _, changed_receipt = stress_rule(changed)
        self.assertEqual(receipt["threshold_dlog_vix"], changed_receipt["threshold_dlog_vix"])
        self.assertTrue(flags.loc[["2022-04-08", "2022-04-15"], "stress"].isna().all())
        self.assertEqual(flags.index.max(), pd.Timestamp("2026-09-25"))
        expected = flags.loc["2010":"2022", "stress_dlog_vix"].dropna().quantile(.9)
        self.assertAlmostEqual(receipt["threshold_dlog_vix"], expected)

    def test_block_bootstrap_preserves_pairs_gaps_and_seed(self):
        dates = pd.date_range("2023-01-06", periods=160, freq="W-FRI")
        stress = (np.arange(len(dates)) % 5 == 0).astype(float)
        frame = pd.DataFrame({"stress": stress, "d_x_govt_raw_bps": 2 * stress,
                              "d_x_rf_raw_bps": stress, "d_us_swap_spread_bps": .5 * stress,
                              "d_foreign_swap_spread_bps": -.5 * stress}, index=dates)
        frame.iloc[33, 1:] = np.nan
        first, n = block_bootstrap(frame, reps=200, seed=11)
        second, _ = block_bootstrap(frame, reps=200, seed=11)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(n, 200)
        np.testing.assert_allclose(first[:, 0, 0], 2.)
        np.testing.assert_allclose(first[:, 1, 0], 0.)
        np.testing.assert_allclose(first[:, 2, 0], 2.)
        with self.assertRaisesRegex(AssertionError, "Keep missing weeks"):
            block_bootstrap(frame.drop(frame.index[33]), reps=10)

    def test_joint_primary_family_and_known_post_signal(self):
        dates = pd.date_range("2010-01-08", "2026-09-25", freq="W-FRI")
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.normal(size=(len(dates), len(FACTORS))), columns=FACTORS)
        frame["week"], frame["valid_change"], frame["cohort"] = dates, True, "fixed8"
        frame["signature"] = np.where(dates < "2022-07-01", "old", "new")
        post = (dates >= "2023-01-01").astype(float)
        frame["d_x_rf_raw_bps"] = rng.normal(size=len(dates))
        frame["d_us_swap_spread_bps"] = rng.normal(size=len(dates))
        frame["d_x_govt_raw_bps"] = (1 + 3 * post) * frame.dlog_vix + rng.normal(size=len(dates))
        frame["d_foreign_swap_spread_bps"] = frame.d_x_rf_raw_bps + frame.d_us_swap_spread_bps - frame.d_x_govt_raw_bps
        weekly = pd.concat([frame.assign(tenor=t) for t in ["5y", "10y"]], ignore_index=True)
        result = joint_tests(weekly)
        primary = result[result.p_holm_primary_family.notna()]
        self.assertEqual(len(primary), 6)
        self.assertTrue(primary.p_holm_primary_family.ge(primary.p_exploratory).all())
        self.assertTrue(primary.p_holm_primary_family.lt(.01).all())
        self.assertEqual(set(result.outcome), set(OUTCOMES))


if __name__ == "__main__":
    unittest.main()
