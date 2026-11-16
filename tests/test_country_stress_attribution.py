import unittest

import numpy as np
import pandas as pd

from src.analyze_country_stress_attribution import (
    COHORT, OUTCOMES, accounting, country_estimates, leave_one_country, make_country_weekly,
    public_overlap_check,
)
from src.analyze_credit_risk_extension import make_weekly


class CountryStressTests(unittest.TestCase):
    def inputs(self):
        dates = pd.bdate_range("2022-12-01", "2023-06-09")
        rows = []
        for i, date in enumerate(dates):
            for c, currency in enumerate(COHORT):
                rf = .1 * (c + 1) * np.sin(i / 3) + .02 * i
                us = .3 * np.cos(i / 6)
                foreign = .2 * (c + 1) * np.cos(i / 4)
                swap = 2 + .01 * i
                for tenor in ["5y", "10y"]:
                    rows.append({"date": date, "currency": currency, "tenor": tenor, "regime": "sofr",
                                 "is_selected_regime_by_public_break": True, "foreign_swap": swap,
                                 "foreign_govt_yield": swap - foreign / 100,
                                 "x_govt_raw_bps": rf + us - foreign, "x_rf_raw_bps": rf,
                                 "us_swap_spread_bps": us, "foreign_swap_spread_bps": foreign})
        market = pd.DataFrame({"vix_index": 20., "spx_index": 4000., "ig_oas": 1.,
                               "hy_oas": 3., "move_index": 80., "bbdxy_index": 1000.}, index=dates)
        weeks = pd.date_range("2022-12-02", "2023-06-09", freq="W-FRI")
        flags = pd.DataFrame({"stress": (np.arange(len(weeks)) % 4 == 0).astype(float),
                              "stress_dlog_vix": .2}, index=weeks)
        return pd.DataFrame(rows), market, flags

    def test_same_dates_weights_units_and_aggregate_reconciliation(self):
        raw, market, flags = self.inputs()
        base = make_weekly(raw, market)
        panel, audit = make_country_weekly(raw, market, base, flags)
        self.assertLess(audit.max_weekly_mean_error_bps.max(), 1e-12)
        self.assertEqual(panel.groupby(["tenor", "week"]).currency.nunique().min(), 8)
        for metric in OUTCOMES:
            country_mean = panel.groupby(["tenor", "week"])[f"d_{metric}"].mean()
            expected = base[base.cohort.eq("fixed8") & base.week.ge("2023-01-01")].set_index(["tenor", "week"])
            expected = expected[f"d_{metric}"].where(expected.valid_change).sort_index()
            np.testing.assert_allclose(country_mean, expected, equal_nan=True)
        np.testing.assert_allclose(panel.foreign_swap_spread_bps, panel.foreign_swap_bps - panel.foreign_govt_yield_bps, atol=1e-12, equal_nan=True)

    def test_missing_country_week_and_regime_switch_do_not_bridge(self):
        raw, market, flags = self.inputs()
        raw = raw[~(raw.currency.eq("AUD") & raw.date.between("2023-02-06", "2023-02-10"))].copy()
        raw.loc[raw.currency.eq("NZD") & raw.date.ge("2023-03-15"), "regime"] = "new_label"
        base = make_weekly(raw, market)
        panel, _ = make_country_weekly(raw, market, base, flags)
        for week in ["2023-02-10", "2023-02-17", "2023-03-17", "2023-03-24"]:
            unavailable = panel[panel.week.eq(week)]
            self.assertFalse(unavailable.valid_change.any())
            self.assertTrue(unavailable.d_x_govt_raw_bps.isna().all())

    def test_country_and_event_accounting_and_deletion_formula(self):
        raw, market, flags = self.inputs()
        base = make_weekly(raw, market)
        panel, _ = make_country_weekly(raw, market, base, flags)
        estimates, events = country_estimates(panel)
        previous = []
        for (cohort, tenor), group in base.groupby(["cohort", "tenor"]):
            sample = group[group.valid_change & group.week.ge("2023-01-01")].set_index("week").join(flags)
            stress = sample[sample.stress.eq(1)].d_x_govt_raw_bps.mean()
            ordinary = sample[sample.stress.eq(0)].d_x_govt_raw_bps.mean()
            for contrast, value in [("stress_mean", stress), ("ordinary_mean", ordinary), ("stress_minus_ordinary", stress - ordinary)]:
                previous.append({"cohort": cohort, "tenor": tenor, "period": "post2023", "outcome": "x_govt_raw_bps",
                                 "contrast": contrast, "estimate_bps": value})
        attribution, ledger = accounting(estimates, pd.DataFrame(previous))
        self.assertLess(ledger.residual_bps.abs().max(), 1e-12)
        np.testing.assert_allclose(attribution.weighted_pair_contribution_bps,
                                   attribution.country_specific_contribution_bps + attribution.common_us_allocation_bps)
        np.testing.assert_allclose(attribution.minus_foreign_spread_contribution_bps,
                                   attribution.minus_foreign_swap_contribution_bps + attribution.plus_foreign_govt_contribution_bps, atol=1e-12)
        omitted = leave_one_country(panel)
        for tenor in ["5y", "10y"]:
            net = ledger[ledger.tenor.eq(tenor) & ledger.contrast.eq("stress_minus_ordinary")].previous_aggregate_bps.item()
            self.assertAlmostEqual(events[events.tenor.eq(tenor)].aggregate_contribution_bps.sum(), net)
            for currency in COHORT:
                pair = attribution[attribution.tenor.eq(tenor) & attribution.currency.eq(currency)].pair_contrast_bps.item()
                actual = omitted[omitted.tenor.eq(tenor) & omitted.omitted_currency.eq(currency)].estimate_bps.item()
                self.assertAlmostEqual(actual, (8 * net - pair) / 7)

    def test_duplicate_and_unshared_us_leg_are_rejected(self):
        raw, market, flags = self.inputs()
        base = make_weekly(raw, market)
        with self.assertRaises(AssertionError):
            make_country_weekly(pd.concat([raw, raw.iloc[[0]]]), market, base, flags)
        changed = raw.copy()
        mask = changed.currency.eq("NZD") & changed.date.ge("2023-02-13")
        changed.loc[mask, "us_swap_spread_bps"] += 1.
        changed.loc[mask, "x_govt_raw_bps"] += 1.
        with self.assertRaises(AssertionError):
            make_country_weekly(changed, market, make_weekly(changed, market), flags)

    def test_public_overlap_is_same_calendar_and_honors_missing_weeks(self):
        raw, market, flags = self.inputs()
        base = make_weekly(raw, market)
        public = raw[['date', 'currency', 'tenor', 'x_govt_raw_bps']].rename(columns={'x_govt_raw_bps': 'cip_govt'})
        public = public[~public.date.between('2023-02-06', '2023-02-10')]
        result = public_overlap_check(raw, market, base, flags, public)
        np.testing.assert_allclose(result.contrast_gap_bps, 0., atol=1e-12)
        np.testing.assert_allclose(result.weekly_change_mae_bps, 0., atol=1e-12)
        for tenor, group in base[base.cohort.eq('fixed8')].groupby('tenor'):
            sample = group[group.valid_change & group.week.ge('2023-01-01')].set_index('week').join(flags)
            sample = sample.drop(index=pd.to_datetime(['2023-02-10', '2023-02-17']), errors='ignore')
            self.assertTrue(result[result.tenor.eq(tenor)].n_stress.eq(sample.stress.eq(1).sum()).all())
            self.assertTrue(result[result.tenor.eq(tenor)].n_ordinary.eq(sample.stress.eq(0).sum()).all())


if __name__ == "__main__":
    unittest.main()
