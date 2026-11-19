import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src import analyze_sovereign_cds as cds


class SovereignCDSTests(unittest.TestCase):
    @unittest.skipUnless(all((cds.INPUT / f).is_file() for f in cds.FOREIGN_WORKBOOKS),
                         "local source workbooks are needed for row-level reconciliation")
    def test_archived_workbooks_agree(self):
        daily, metadata = cds.load_cds(cds.INPUT)
        self.assertEqual(len(daily), 12111)
        self.assertEqual(len(metadata), 16)
        self.assertNotIn("USD", daily.currency.unique())

    def test_relative_slope_is_invariant_with_week_effects(self):
        rng = np.random.default_rng(5)
        weeks = pd.date_range("2023-01-06", periods=35, freq="W-FRI")
        rows = []
        us = []
        for i, week in enumerate(weeks):
            us_value = 30 + rng.normal()
            for date in pd.bdate_range(week - pd.Timedelta(days=4), week):
                us.append((date, "5y", us_value))
            for country in ("AUD", "CAD", "CHF"):
                foreign = 30 + rng.normal()
                swap = 200 + rng.normal()
                spread = 0.3 * foreign + us_value + rng.normal(scale=0.2)
                for date in pd.bdate_range(week - pd.Timedelta(days=4), week):
                    if country == "CHF" and i == 5:
                        continue
                    rows.append((date, country, "5y", "a", foreign, spread, swap + spread, swap))
        daily = pd.DataFrame(rows, columns=cds.KEY + ["regime", "cds", *cds.OUTCOMES])
        us = pd.DataFrame(us, columns=["date", "tenor", "us_cds"])
        market = pd.DataFrame(rng.normal(size=(len(weeks), len(cds.FACTORS))), columns=cds.FACTORS)
        market["week"] = weeks
        reg, matched = cds.relative_regressions(daily, market, us)
        self.assertFalse(matched.empty)
        abs_beta = reg.query("spec == 'country_week_FE_absolute_matched'").beta
        rel_beta = reg.query("spec == 'country_week_FE_relative'").beta
        np.testing.assert_allclose(abs_beta, rel_beta, atol=1e-9, rtol=0)
        self.assertEqual(set(reg.query("spec == 'market_two_cds_legs'").factor),
                         {*cds.FACTORS, "d_cds", "d_us_cds"})

    def test_us_input_rejects_duplicate_quotes(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "us.csv"
            path.write_text("date,tenor,us_cds\n2023-01-05,5y,20\n2023-01-05,5y,21\n")
            with self.assertRaisesRegex(ValueError, "unique daily quotes"):
                cds.load_us(path)
