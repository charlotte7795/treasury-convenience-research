"""Fixed-vintage gross-debt screen; not a net marketable supply test."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analyze_post_meeting_research import weekly_hac
from src.analyze_treasury_interim import CLEAN, COHORT, OFFICIAL, TABLE

EXPOSURE = OFFICIAL / "imf_fiscal_monitor_oct2022_debt_exposure.csv"
PUBLIC_PANEL = OFFICIAL.parents[1] / "outputs/samim-stress-supply/sovereign_stress_supply_public_panel.csv"
AOFM_ANCHOR = OFFICIAL / "stress_supply_2026-11-03/aofm_fy2022_treasury_bonds.csv"
TARIFF_WEEKS = ["2025-04-04", "2025-04-11"]
OUTCOMES = ["d_x_govt_raw_bps", "d_x_rf_raw_bps", "d_us_swap_spread_bps", "d_foreign_swap_spread_bps"]
PAIRED = ("CAD", "GBP", "NZD", "SEK")


def screen(panel, exposure):
    assert exposure.currency.is_unique and set(exposure.currency) == set(COHORT)
    assert exposure.gross_debt_pct_gdp.gt(0).all()
    panel = panel[panel.valid_change & panel.stress.notna()].copy()
    assert panel.week.min() > pd.Period(exposure.vintage_month.max(), freq="M").end_time
    return weekly_screen(panel, exposure.set_index("currency").gross_debt_pct_gdp.reindex(COHORT))


def stress_contrast(values, stress):
    fit = sm.OLS(values.loc[stress.index], sm.add_constant(stress)).fit()
    beta = fit.params["stress"]
    se = np.sqrt(weekly_hac(fit, stress.index)[1, 1])
    return beta, beta - 1.96 * se, beta + 1.96 * se


def weekly_screen(panel, exposure, *, episodes=False):
    cohort = exposure.index.tolist()
    panel = panel.loc[panel.currency.isin(cohort)]
    rows = []
    for tenor, group in panel.groupby("tenor"):
        flags = group.groupby("week").stress.first()
        assert group.groupby("week").stress.nunique().eq(1).all()
        data = group.pivot(index="week", columns="currency", values=OUTCOMES)
        assert data.notna().all().all()
        for omitted in ["none", *cohort]:
            currencies = [c for c in cohort if c != omitted]
            x = exposure.loc[currencies].to_numpy() / 10
            x -= x.mean()
            assert x @ x > 0
            # A single cross-sectional slope per week avoids treating the eight pairs as independent shocks.
            slopes = pd.DataFrame({m: data[m][currencies].to_numpy() @ x / (x @ x) for m in OUTCOMES}, index=data.index)
            np.testing.assert_allclose(slopes[OUTCOMES[0]], slopes[OUTCOMES[1]] + slopes[OUTCOMES[2]] - slopes[OUTCOMES[3]], atol=1e-10)
            windows = [("all", flags.notna()), ("ex_tariff", ~flags.index.isin(pd.to_datetime(TARIFF_WEEKS)))]
            if episodes and omitted == "none":
                dates = flags.index[flags.eq(1)]
                blocks = dates.to_series().diff().ne(pd.Timedelta(days=7)).cumsum()
                windows += [(f"ex_episode_{b.index[0].date()}", ~flags.index.isin(b.index)) for _, b in blocks.groupby(blocks)]
            for sample, keep in windows:
                stress = flags.loc[keep]
                metrics = OUTCOMES if omitted == "none" and not sample.startswith("ex_episode_") else OUTCOMES[:1]
                for metric in metrics:
                    beta, low, high = stress_contrast(slopes[metric], stress)
                    rows.append({"tenor": tenor, "sample": sample, "omitted": omitted, "outcome": metric,
                                 "n_stress": int(stress.sum()), "n_ordinary": int(stress.eq(0).sum()),
                                 "bps_per_10pp": beta, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def german_general_government():
    """Matched face value, broader than federal-only Bund supply."""
    data_dir = AOFM_ANCHOR.parent
    holdings = pd.read_csv(data_dir / "ecb_shss_de_long_gov_face.csv")
    gross = pd.read_csv(data_dir / "ecb_csec_de_long_gov_face.csv")
    for data in (holdings, gross):
        for field, value in {"UNIT_MULT": 6, "INSTR_ASSET": "F3", "MATURITY": "L",
                             "CURRENCY_DENOM": "_T", "VALUATION": "F", "STO": "LE"}.items():
            assert data[field].eq(value).all()
    assert holdings[["REF_AREA", "REF_SECTOR", "COUNTERPART_AREA", "COUNTERPART_SECTOR"]].eq(["U2", "S121", "DE", "S13"]).all().all()
    assert gross[["REF_AREA", "REF_SECTOR", "UNIT_MEASURE"]].eq(["DE", "S13", "EUR"]).all().all()
    holdings.index = pd.PeriodIndex(holdings.TIME_PERIOD, freq="Q").end_time.normalize()
    gross.index = pd.PeriodIndex(gross.TIME_PERIOD, freq="M").end_time.normalize()
    matched = pd.DataFrame({
        "outstanding_lcu_bn": gross.OBS_VALUE.reindex(holdings.index) / 1000,
        "cb_held_lcu_bn": holdings.OBS_VALUE / 1000,
        "holdings_obs_status": holdings.OBS_STATUS,
    }).rename_axis("quarter_end")
    assert len(matched) == 19 and matched.index.is_unique and matched.notna().all().all()
    assert matched.cb_held_lcu_bn.ge(0).all() and matched.cb_held_lcu_bn.lt(matched.outstanding_lcu_bn).all()
    return matched


def paired_diagnostic(panel, public, imf, aofm, germany):
    """Fixed pre-sample observation dates; most value vintages are unverified."""
    valid = panel.loc[panel.valid_change & panel.stress.notna()]
    assert valid.groupby(["tenor", "week"]).currency.nunique().eq(len(COHORT)).all()
    initial = public.set_index(["currency", "quarter_end"], verify_integrity=True)
    stock = aofm.set_index(["currency", "quarter_end"], verify_integrity=True).outstanding_lcu_bn
    prior = initial.loc[stock.index, "outstanding_lcu_bn"]
    np.testing.assert_allclose(prior.dropna(), stock.loc[prior.dropna().index])
    initial.loc[stock.index, "outstanding_lcu_bn"] = stock
    germany = germany.assign(currency="EUR").reset_index().set_index(["currency", "quarter_end"])
    amounts = ["outstanding_lcu_bn", "cb_held_lcu_bn"]
    assert initial.loc[germany.index, amounts].isna().all().all()
    initial.loc[germany.index, amounts] = germany[amounts]
    fields = ["imf_gross_pct_gdp", "gross_market_pct_gdp", "net_market_pct_gdp"]
    exposures, estimates = [], []
    for label, baseline, cohort in [
        ("2021Q4_four", "2021-12-31", PAIRED),
        ("2022Q2_four", "2022-06-30", PAIRED),
        ("2022Q2_plus_AUD", "2022-06-30", ("AUD", *PAIRED)),
        ("2022Q2_plus_AUD_DE", "2022-06-30", ("AUD", "EUR", *PAIRED)),
    ]:
        date = pd.Timestamp(baseline)
        assert date < valid.week.min()
        paired = initial.xs(date, level="quarter_end").loc[list(cohort)].copy()
        assert paired[amounts + ["nominal_gdp_4q_lcu_bn"]].notna().all().all()
        assert paired.cb_held_lcu_bn.ge(0).all() and paired.cb_held_lcu_bn.le(paired.outstanding_lcu_bn).all()
        paired["gross_market_pct_gdp"] = 100 * paired.outstanding_lcu_bn / paired.nominal_gdp_4q_lcu_bn
        paired["net_market_pct_gdp"] = 100 * (paired.outstanding_lcu_bn - paired.cb_held_lcu_bn) / paired.nominal_gdp_4q_lcu_bn
        comparable = paired.net_supply_pct_gdp.notna()
        np.testing.assert_allclose(paired.loc[comparable, "net_market_pct_gdp"], paired.loc[comparable, "net_supply_pct_gdp"])
        paired["imf_gross_pct_gdp"] = imf.set_index("currency").loc[list(cohort), "gross_debt_pct_gdp"]
        exposures.append(paired[fields].assign(design=label, quarter_end=date).reset_index())
        for field in fields:
            results = weekly_screen(valid, paired[field], episodes=True).assign(design=label, exposure=field)
            # Compare proxy magnitudes on a common scale, using the full design cohort.
            results["bps_per_full_cohort_sd"] = results.bps_per_10pp * paired[field].std(ddof=0) / 10
            estimates.append(results)
    estimates = pd.concat(estimates, ignore_index=True)
    assert estimates.groupby(["tenor", "sample"])[["n_stress", "n_ordinary"]].nunique().eq(1).all().all()
    return pd.concat(exposures, ignore_index=True), estimates


def standardized_net_advantage(panel, exposures, estimates):
    """Compare net and gross slopes on the same per-SD scale and weeks."""
    panel = panel.loc[panel.valid_change & panel.stress.notna()]
    rows = []
    for design, baseline in exposures.groupby("design"):
        fields = ["net_market_pct_gdp", "gross_market_pct_gdp"]
        x = baseline.set_index("currency")[fields]
        z = (x - x.mean()) / x.std(ddof=0)
        weights = z[fields[0]].sub(z[fields[1]]) / len(x)
        for tenor, group in panel.groupby("tenor"):
            gov = group.pivot(index="week", columns="currency", values=OUTCOMES[0])
            differences = gov[weights.index] @ weights
            flags = group.groupby("week").stress.first()
            for sample, stress in (("all", flags), ("ex_tariff", flags.drop(pd.to_datetime(TARIFF_WEEKS)))):
                beta, low, high = stress_contrast(differences, stress)
                fits = estimates.loc[
                    estimates.design.eq(design) & estimates.tenor.eq(tenor) & estimates["sample"].eq(sample)
                    & estimates.omitted.eq("none") & estimates.outcome.eq(OUTCOMES[0])
                ].set_index("exposure").bps_per_full_cohort_sd
                np.testing.assert_allclose(beta, fits[fields[0]] - fits[fields[1]], atol=1e-10)
                rows.append(dict(design=design, tenor=tenor, sample=sample, bps_net_minus_gross=beta,
                                 ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def run_paired_diagnostic():
    panel = pd.read_csv(CLEAN / "country_stress_weekly_panel.csv", parse_dates=["week"])
    public = pd.read_csv(PUBLIC_PANEL, parse_dates=["quarter_end"])
    germany = german_general_government()
    german_gdp = public.query("currency == 'EUR'").set_index("quarter_end").nominal_gdp_4q_lcu_bn
    german_data = germany.join(german_gdp, validate="1:1")
    german_data["net_supply_pct_gdp"] = 100 * (german_data.outstanding_lcu_bn - german_data.cb_held_lcu_bn) / german_data.nominal_gdp_4q_lcu_bn
    german_data.to_csv(TABLE / "stress_supply_germany_general_government.csv")
    exposures, results = paired_diagnostic(
        panel, public, pd.read_csv(EXPOSURE), pd.read_csv(AOFM_ANCHOR, parse_dates=["quarter_end"]), germany,
    )
    exposures.to_csv(TABLE / "stress_supply_paired_exposures.csv", index=False)
    results.to_csv(TABLE / "stress_supply_paired_diagnostic.csv", index=False)
    advantage = standardized_net_advantage(panel, exposures, results)
    advantage.to_csv(TABLE / "stress_supply_net_minus_gross.csv", index=False)
    return exposures, results, advantage


def main():
    panel = pd.read_csv(CLEAN / "country_stress_weekly_panel.csv", parse_dates=["week"])
    results = screen(panel, pd.read_csv(EXPOSURE))
    results.to_csv(TABLE / "stress_supply_gross_debt_pilot.csv", index=False)
    return results


if __name__ == "__main__":
    print(main().query("omitted == 'none'").round(4).to_string(index=False))
