import unittest

import pandas as pd

from src.analyze_stress_supply_pilot import (
    AOFM_ANCHOR, CLEAN, COHORT, EXPOSURE, OUTCOMES, PUBLIC_PANEL,
    german_general_government, paired_diagnostic, screen, standardized_net_advantage,
)


class StressSupplyPilotTests(unittest.TestCase):
    def test_recovers_stress_slope_without_common_weekly_term(self):
        exposure = pd.read_csv(EXPOSURE)
        debt = exposure.set_index("currency").gross_debt_pct_gdp / 10
        rows = []
        for j, week in enumerate(pd.date_range("2025-03-07", periods=40, freq="W-FRI")):
            stress = int(j % 4 == 0 or str(week.date()) in {"2025-04-04", "2025-04-11"})
            for i, currency in enumerate(COHORT):
                gov = (1 + 2 * stress) * debt[currency] + 0.01 * i * j + j
                rows.append(dict(week=week, tenor="5y", currency=currency, stress=stress, valid_change=True,
                                 **dict(zip(OUTCOMES, [gov, 0., float(j), float(j) - gov]))))
        panel = pd.DataFrame(rows)
        fits = screen(panel, exposure)
        full = fits.query("sample == 'all' and omitted == 'none'").set_index("outcome")
        self.assertAlmostEqual(full.loc[OUTCOMES[0], "bps_per_10pp"],
                               -full.loc[OUTCOMES[3], "bps_per_10pp"])
        self.assertAlmostEqual(full.loc[OUTCOMES[2], "bps_per_10pp"], 0.)
        self.assertGreater(full.loc[OUTCOMES[0], "bps_per_10pp"], 1.9)
        self.assertLess(full.loc[OUTCOMES[0], "bps_per_10pp"], 2.1)
        reduced = fits.query("sample == 'ex_tariff' and omitted == 'none'").set_index("outcome")
        self.assertEqual(full.n_stress.iloc[0] - reduced.n_stress.iloc[0], 2)
        with self.assertRaises(AssertionError):
            screen(panel, exposure.assign(vintage_month="2026-10"))

    def test_paired_design_keeps_calendars_and_separates_german_scope(self):
        panel = pd.read_csv(CLEAN / "country_stress_weekly_panel.csv", parse_dates=["week"])
        public = pd.read_csv(PUBLIC_PANEL, parse_dates=["quarter_end"])
        anchor = pd.read_csv(AOFM_ANCHOR, parse_dates=["quarter_end"])
        german = german_general_government()
        exposures, results = paired_diagnostic(panel, public, pd.read_csv(EXPOSURE), anchor, german)
        advantage = standardized_net_advantage(panel, exposures, results)
        self.assertEqual(len(german), 19)
        self.assertEqual(exposures.groupby("design").currency.nunique().to_dict(), {
            "2021Q4_four": 4, "2022Q2_four": 4, "2022Q2_plus_AUD": 5, "2022Q2_plus_AUD_DE": 6,
        })
        net = results.loc[results.outcome.eq(OUTCOMES[0]) & results.exposure.eq("net_market_pct_gdp")]
        reference = net.query("design == '2022Q2_plus_AUD_DE' and omitted == 'none'").set_index(["tenor", "sample"])
        self.assertEqual(reference.loc[("5y", "all"), "n_stress"], 13)
        self.assertEqual(reference.loc[("5y", "ex_tariff"), "n_stress"], 11)
        self.assertEqual(reference.loc[("5y", "ex_episode_2025-04-04"), "n_stress"], 11)
        self.assertAlmostEqual(reference.loc[("5y", "ex_tariff"), "bps_per_10pp"], 0.5974, places=3)
        self.assertEqual(len(advantage), 16)
        self.assertTrue((advantage.ci_low.lt(0) & advantage.ci_high.gt(0)).all())
