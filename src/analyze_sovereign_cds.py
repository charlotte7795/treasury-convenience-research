"""Local sovereign CDS extension. Pass the existing repo and the downloaded workbooks."""
import argparse
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/refresh/sovereign_cds_2026-11-19"
FOREIGN_WORKBOOKS = ("Bloomberg_8_Country_5Y_10Y_Sovereign_CDS_BDH_Template.xlsx",
                     "Sovereign_CDS_5Y_10Y_Complete_Extracted.xlsx")
KEY = ["date", "currency", "tenor"]
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-09-30")
LAST_FRIDAY = pd.Timestamp("2026-09-25")
OUTCOMES = ["bond_minus_swap", "bond_yield", "swap_rate"]
TENORS = ["5y", "10y"]
CDS = "cds"
FACTORS = ["dlog_vix", "spx_return", "d_ig_bps", "d_hy_bps"]
MARKET = ["vix_index", "spx_index", "ig_oas", "hy_oas"]


def load_cds(folder):
    workbook, extraction = [folder / name for name in FOREIGN_WORKBOOKS]
    if not (workbook.is_file() and extraction.is_file()):
        raise FileNotFoundError(f"Place the two local CDS workbooks described in {folder / 'README.md'} before rerunning.")
    wb = load_workbook(workbook, read_only=True, data_only=True)
    data, metadata = [], []
    for sheet in wb:
        rows = list(sheet.values)
        if not str(rows[0][0]).endswith("SOVEREIGN CDS"):
            continue
        currency = rows[2][1]
        tenor = "10y" if sheet.title.endswith("10Y") else "5y"
        metadata.append((currency, tenor, rows[3][1], rows[4][1]))
        for row in rows:
            if isinstance(row[0], date) and isinstance(row[1], (int, float)):
                data.append((pd.Timestamp(row[0]), currency, tenor, float(row[1])))
    wb.close()
    cds = pd.DataFrame(data, columns=KEY + [CDS])
    meta = pd.DataFrame(metadata, columns=["currency", "tenor", "country", "ticker"])
    extracted = pd.read_excel(extraction,
                              sheet_name="LONG_DATA", usecols=[0, 2, 3, 4])
    extracted.columns = KEY + ["pdf_cds"]
    extracted.tenor = extracted.tenor.str.lower()
    comparison = cds.merge(extracted, on=KEY, how="outer", indicator=True, validate="one_to_one")
    assert comparison._merge.eq("both").all()
    assert comparison.cds.eq(comparison.pdf_cds).all()
    assert len(meta) == 16 and cds.cds.gt(0).all() and cds.date.between(START, END).all()
    return cds, meta


def load_repo(repo):
    rates = pd.read_csv(repo / "data/derived/raw_g10_benchmark_eq6_5y10y_refreshed_panel.csv.gz",
                        parse_dates=["date"], low_memory=False)
    rates = rates.loc[rates.is_selected_regime_by_public_break & rates.date.between(START, END),
                      KEY + ["regime", "foreign_govt_yield", "foreign_swap", "foreign_swap_spread_bps"]].copy()
    rates["bond_yield"] = 100 * rates.foreign_govt_yield
    rates["swap_rate"] = 100 * rates.foreign_swap
    rates["bond_minus_swap"] = rates.bond_yield - rates.swap_rate
    assert np.allclose(rates.bond_minus_swap, -rates.foreign_swap_spread_bps, atol=1e-8)
    market_file = repo / "data/refresh/risk_extension_2026-11-02/Risk_Extension_Bloomberg_BDH_Template.xlsx"
    market = []
    wb = load_workbook(market_file, data_only=True, read_only=True)
    for sheet, _, variable, *_ in wb["01_Inputs"].iter_rows(min_row=2, max_row=7, values_only=True):
        if variable not in MARKET:
            continue
        rows = [(dt, quote) for dt, quote, *_ in wb[sheet].iter_rows(min_row=5, values_only=True)
                if isinstance(dt, date) and isinstance(quote, (int, float))]
        market.append(pd.DataFrame(rows, columns=["date", variable]).set_index("date"))
    wb.close()
    daily = pd.concat(market, axis=1).dropna().sort_index().loc[START:LAST_FRIDAY]
    calendar = daily.resample("W-FRI")
    weekly = calendar.mean()[MARKET].where(calendar.size().ge(3))
    weekly[MARKET[:2]] = np.log(weekly[MARKET[:2]])
    weekly[MARKET[2:]] *= 100
    weekly = weekly.diff().set_axis(FACTORS, axis=1).rename_axis("week").reset_index()
    return rates[KEY + ["regime", *OUTCOMES]], weekly


def aggregate_weeks(daily, min_days=3):
    weeks = []
    columns = [CDS] + OUTCOMES
    for (currency, tenor), group in daily[daily.date.le(LAST_FRIDAY)].groupby(["currency", "tenor"]):
        calendar = group.set_index("date").resample("W-FRI")
        levels = calendar[columns].mean()
        count = calendar.size()
        regime = calendar.regime.first()
        good = count.ge(min_days) & calendar.regime.nunique().eq(1)
        levels = levels.where(good)
        valid_change = good & good.shift(fill_value=False) & regime.eq(regime.shift())
        diff = levels.diff().where(valid_change).add_prefix("d_")
        block = levels.join(diff).assign(currency=currency, tenor=tenor, joint_days=count,
                                         valid_level=good, valid_change=valid_change)
        weeks.append(block.rename_axis("week").reset_index())
    return pd.concat(weeks, ignore_index=True)


def residualize(frame, columns, time_fe):
    effects = pd.get_dummies(frame.currency, dtype=float).to_numpy()
    if time_fe:
        dates = pd.get_dummies(frame.week, dtype=float).iloc[:, 1:].to_numpy()
        effects = np.column_stack((effects, dates))
    matrix = frame[columns].to_numpy()
    params, _, rank, _ = np.linalg.lstsq(effects, matrix, rcond=None)
    return matrix - effects @ params, rank


def fit(frame, predictors, time_fe):
    # Frisch–Waugh–Lovell. Compute HAC on calendar-week summed scores so
    # same-week cross-country dependence and serial dependence are allowed.
    if time_fe:
        frame = frame[frame.groupby("week").currency.transform("size").ge(2)]
    ycols = ["d_" + name for name in OUTCOMES]
    data, fe_rank = residualize(frame, predictors + ycols, time_fe)
    x, y = data[:, :len(predictors)], data[:, len(predictors):]
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    assert rank == len(predictors) and len(x) > fe_rank + rank
    residuals = y - x @ coefficients
    rss = (residuals ** 2).sum(axis=0)
    weeks = pd.DatetimeIndex(frame.week).to_period("W-FRI").asi8
    bread = np.linalg.inv(x.T @ x)
    rows = []
    for i, outcome in enumerate(OUTCOMES):
        score = np.zeros((weeks.max() - weeks.min() + 1, len(predictors)))
        np.add.at(score, weeks - weeks.min(), x * residuals[:, i, None])
        for lag_count in [4, 13]:
            meat = score.T @ score
            for lag in range(1, lag_count + 1):
                cross = score[lag:].T @ score[:-lag]
                meat += (1 - lag / (lag_count + 1)) * (cross + cross.T)
            cov = bread @ meat @ bread * len(x) / (len(x) - fe_rank - rank)
            se = np.sqrt(np.maximum(0, cov.diagonal()))
            for j, factor in enumerate(predictors):
                beta = coefficients[j, i]
                rows.append(dict(outcome=outcome, factor=factor, beta=beta, se=se[j], hac_weeks=lag_count,
                                 ci_low=beta - 1.96 * se[j], ci_high=beta + 1.96 * se[j],
                                 p_value=2 * norm.sf(abs(beta / se[j])),
                                 within_r2=1 - rss[i] / (y[:, i] ** 2).sum(), rss=rss[i],
                                 n=len(x), countries=frame.currency.nunique(), weeks=frame.week.nunique(),
                                 first_week=frame.week.min(), last_week=frame.week.max()))
    assert np.allclose(coefficients[:, 0], coefficients[:, 1] - coefficients[:, 2], atol=1e-10)
    return pd.DataFrame(rows)


def regressions(weekly, weekly_four):
    records = []
    required = ["d_cds", *["d_" + c for c in OUTCOMES]]

    def append(frame, predictors, time_fe, sample, tenor, spec):
        records.append(fit(frame, predictors, time_fe).assign(sample=sample, tenor=tenor, spec=spec))

    complete = weekly.dropna(subset=required)
    for tenor in TENORS:
        group = complete[complete.tenor.eq(tenor)]
        append(group, ["d_cds"], True, "main", tenor, "country_week_FE")
        matched = group.dropna(subset=FACTORS)
        append(matched, FACTORS, False, "main", tenor, "market_baseline")
        append(matched, FACTORS + ["d_cds"], False, "main", tenor, "market_plus_CDS")
        for currency in sorted(group.currency.unique()):
            append(group[group.currency.ne(currency)], ["d_cds"], True, "omit_" + currency,
                   tenor, "country_week_FE")
        append(group[group.week.ge("2025-01-03")], ["d_cds"], True, "2025_onward", tenor, "country_week_FE")
        common = group[group.week.ge("2025-07-18")]
        common = common[common.groupby("week").currency.transform("size").eq(8)]
        append(common, ["d_cds"], True, "balanced8_since_Jul2025", tenor, "country_week_FE")
        early = group[group.groupby("currency").week.transform("min").lt("2023-03-01")]
        append(early, ["d_cds"], True, "early_coverage", tenor, "country_week_FE")
        four = weekly_four[weekly_four.tenor.eq(tenor)].dropna(subset=required)
        append(four, ["d_cds"], True, "four_joint_days", tenor, "country_week_FE")
    return pd.concat(records, ignore_index=True)


def monthly(frame, columns):
    block = frame.assign(month=frame.date.dt.to_period("M"))
    grouped = block.groupby(["currency", "tenor", "month"])
    result = grouped[columns].mean().where(grouped.size().ge(15) & grouped.regime.nunique().eq(1))
    return result.reset_index()


def compare_windows(daily, rates, metadata, reg):
    columns = [CDS] + OUTCOMES
    matched = monthly(daily, columns)
    full = monthly(rates, OUTCOMES)
    coverage = metadata.copy()
    keys = ["currency", "tenor"]
    for label, first, last in [("2023", "2023-01", "2023-12"), ("2026", "2026-01", "2026-09")]:
        for tag, months, cols in [("matched", matched, columns), ("full_rates", full, OUTCOMES)]:
            sub = months[months.month.between(pd.Period(first), pd.Period(last))].groupby(keys)
            levels = sub[cols].mean().add_prefix(f"{tag}_{label}_")
            levels[f"{tag}_{label}_months"] = sub[cols[0]].count()
            coverage = coverage.merge(levels.reset_index(), how="left", on=keys, validate="one_to_one")
    for tag, cols in [("matched", columns), ("full_rates", OUTCOMES)]:
        for col in cols:
            coverage[f"{tag}_change_{col}"] = coverage[f"{tag}_2026_{col}"] - coverage[f"{tag}_2023_{col}"]
        for year in [2023, 2026]:
            entry = f"{tag}_{year}_months"
            coverage[entry] = coverage[entry].fillna(0).astype(int)
    stage = reg.query("sample == 'main' and spec == 'country_week_FE' and factor == 'd_cds' and hac_weeks == 4")
    for outcome in OUTCOMES:
        slopes = stage[stage.outcome.eq(outcome)].set_index("tenor")
        for label, coefficient in [("fitted", "beta"), ("ci_low", "ci_low"), ("ci_high", "ci_high")]:
            coverage[f"{label}_change_{outcome}"] = coverage.tenor.map(slopes[coefficient]) * coverage.matched_change_cds
    for outcome in OUTCOMES:
        lo, hi = [coverage[f"ci_{bound}_change_{outcome}"].copy() for bound in ["low", "high"]]
        coverage[f"ci_low_change_{outcome}"], coverage[f"ci_high_change_{outcome}"] = np.minimum(lo, hi), np.maximum(lo, hi)
    coverage["full_window"] = coverage.matched_2023_months.eq(12) & coverage.matched_2026_months.eq(9)
    coverage["near_full_window"] = coverage.matched_2023_months.ge(11) & coverage.matched_2026_months.eq(9)
    coverage["fitted_repo_foreign_spread_change"] = -coverage.fitted_change_bond_minus_swap
    coverage["actual_repo_foreign_spread_change"] = -coverage.matched_change_bond_minus_swap
    return coverage


def attribution_summary(rebound):
    rows = []
    for tenor in TENORS:
        group = rebound[rebound.tenor.eq(tenor)]
        for sample in ["full_window", "near_full_window"]:
            eligible = group[group[sample]]
            actual = eligible.matched_change_bond_minus_swap.mean()
            fitted = eligible.fitted_change_bond_minus_swap.mean()
            rows.append(dict(tenor=tenor, sample=sample, currencies=" ".join(sorted(eligible.currency)),
                             n_countries=len(eligible), average_cds_change=eligible.matched_change_cds.mean(),
                             actual_bond_minus_swap_change=actual, fitted_bond_minus_swap_change=fitted,
                             fitted_ci_low=eligible.ci_low_change_bond_minus_swap.mean(),
                             fitted_ci_high=eligible.ci_high_change_bond_minus_swap.mean(),
                             signed_share_pct=100*fitted/actual, full8_actual_bond_minus_swap_change=
                             group.full_rates_change_bond_minus_swap.mean()))
    return pd.DataFrame(rows)


def plot_rebound(frame, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), layout="constrained")
    for tenor, ax in zip(TENORS, axes):
        sub = frame[frame.tenor.eq(tenor)].sort_values("currency")
        x = np.arange(len(sub))
        ax.bar(x-.23, sub.full_rates_change_bond_minus_swap, .23, color="#b7c8de", label="Actual (all rate dates)")
        use = sub.near_full_window.to_numpy()
        ax.bar(x[use], sub.loc[use, "matched_change_bond_minus_swap"], .23, color="#214a75", label="Actual on CDS-matched dates")
        ax.bar(x[use]+.23, sub.loc[use, "fitted_change_bond_minus_swap"], .23, color="#de9251", label="CDS-implied on matched dates")
        name = [ccy + ("*" if near and not full else "†" if not near else "")
                for ccy, near, full in zip(sub.currency, sub.near_full_window, sub.full_window)]
        ax.set_xticks(x, name)
        ax.set_title(tenor.upper(), loc="left", fontweight="bold")
        ax.set_ylabel("Yield − swap (bps)")
        ax.axhline(0, c="#556273", lw=.8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("How much of the foreign spread change moves with sovereign CDS?\n2026 Jan–Sep average minus the 2023 average", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("* 11 of 12 baseline months.   † Insufficient CDS coverage for the comparison.")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def load_us(path):
    if not path.exists():
        return pd.DataFrame(columns=["date", "tenor", "us_cds"])
    frame = pd.read_csv(path, parse_dates=["date"])[["date", "tenor", "us_cds"]]
    frame["tenor"] = frame.tenor.str.lower()
    if (frame.empty or frame.date.isna().any() or not frame.tenor.isin(TENORS).all()
            or frame.duplicated(["date", "tenor"]).any()
            or not np.isfinite(frame.us_cds).all() or not frame.us_cds.gt(0).all()):
        raise ValueError("US CDS requires unique daily quotes, 5y/10y, positive spread bps")
    return frame[frame.date.between(START, END) & frame.date.dt.weekday.lt(5)]


def relative_regressions(daily, market, us):
    records, calendar = [], []
    for tenor in TENORS:
        quotes = us[us.tenor.eq(tenor)]
        if quotes.empty:
            continue
        group = daily[daily.tenor.eq(tenor)].merge(quotes, on=["date", "tenor"], validate="many_to_one")
        if group.empty:
            continue
        grouped = quotes[quotes.date.le(LAST_FRIDAY)].set_index("date").us_cds.resample("W-FRI")
        us_week = grouped.mean().where(grouped.size().ge(3)).diff().rename("d_us_cds")
        weekly = aggregate_weeks(group).merge(us_week, left_on="week", right_index=True,
                                              validate="many_to_one")
        sample = weekly[weekly.valid_change].dropna(subset=["d_us_cds"]).copy()
        if sample.week.nunique() < 26 or sample.currency.nunique() < 2:
            continue
        sample["d_relative_cds"] = sample.d_cds - sample.d_us_cds
        # One US series per tenor and week is absorbed by the week effects.
        abs_fit = fit(sample, ["d_cds"], True)
        rel_fit = fit(sample, ["d_relative_cds"], True)
        assert np.allclose(abs_fit[["beta", "se"]], rel_fit[["beta", "se"]], atol=1e-9, rtol=0)
        records.extend([abs_fit.assign(tenor=tenor, spec="country_week_FE_absolute_matched"),
                        rel_fit.assign(tenor=tenor, spec="country_week_FE_relative")])
        controlled = sample.merge(market, on="week", how="left", validate="many_to_one").dropna(subset=FACTORS)
        if controlled.week.nunique() >= 26:
            for label, columns in [
                ("market_baseline_matched", FACTORS),
                ("market_absolute_matched", FACTORS + ["d_cds"]),
                ("market_relative", FACTORS + ["d_relative_cds"]),
                ("market_two_cds_legs", FACTORS + ["d_cds", "d_us_cds"]),
            ]:
                records.append(fit(controlled, columns, False).assign(tenor=tenor, spec=label))
        calendar.append(group.assign(relative_cds=group.cds - group.us_cds))
    results = (pd.concat(records, ignore_index=True) if records else
               pd.DataFrame(columns=["tenor", "spec", "outcome", "factor", "n", "weeks",
                                     "beta", "ci_low", "ci_high", "within_r2", "hac_weeks"]))
    matched = pd.concat(calendar, ignore_index=True) if calendar else pd.DataFrame()
    return results, matched


def relative_rebound(rebound, relative, matched):
    rows = []
    if not matched.empty:
        columns = ["cds", "us_cds", "relative_cds", "bond_minus_swap"]
        levels = monthly(matched, columns)
        slopes = relative.query("spec == 'country_week_FE_relative' and outcome == 'bond_minus_swap' and hac_weeks == 4")
        for (currency, tenor), group in levels.groupby(["currency", "tenor"]):
            before = group[group.month.dt.year.eq(2023)][columns].dropna()
            after = group[group.month.between(pd.Period("2026-01"), pd.Period("2026-09"))][columns].dropna()
            change = after.mean() - before.mean()
            beta = slopes.loc[slopes.tenor.eq(tenor), "beta"].item()
            rows.append(dict(currency=currency, tenor=tenor, months_2023=len(before), months_2026=len(after),
                             foreign_cds_change=change.cds, us_cds_change=change.us_cds,
                             relative_cds_change=change.relative_cds, actual_change=change.bond_minus_swap,
                             relative_fitted=beta * change.relative_cds, absolute_fitted=beta * change.cds))
    country = pd.DataFrame(rows, columns=["currency", "tenor", "months_2023", "months_2026",
                                          "foreign_cds_change", "us_cds_change", "relative_cds_change",
                                          "actual_change", "relative_fitted", "absolute_fitted"])
    summary = []
    for tenor in TENORS:
        strict = rebound[rebound.full_window & rebound.tenor.eq(tenor)]
        beta = (strict.fitted_change_bond_minus_swap / strict.matched_change_cds).mean()
        foreign = strict.matched_change_cds.mean()
        actual = strict.matched_change_bond_minus_swap.mean()
        eligible = country[country.tenor.eq(tenor) & country.months_2023.eq(12) & country.months_2026.eq(9)]
        means = eligible[["foreign_cds_change", "us_cds_change", "relative_cds_change",
                          "actual_change", "relative_fitted", "absolute_fitted"]].mean()
        summary.append(dict(tenor=tenor, n_countries=len(eligible),
                            **means.to_dict(), reference_foreign_cds_change=foreign, reference_actual_change=actual,
                            us_change_for_zero_fitted=foreign,
                            us_change_to_match_observed=foreign-actual/beta))
    return country, pd.DataFrame(summary)


def run(repo=ROOT, cds_inputs=INPUT, output=None, us_path=None):
    output = Path(output) if output else repo / "output/tables"
    output.mkdir(parents=True, exist_ok=True)
    us_path = Path(us_path) if us_path else cds_inputs / "us_sovereign_cds.csv"
    cds, metadata = load_cds(cds_inputs)
    rates, market = load_repo(repo)
    daily = cds.merge(rates, on=KEY, validate="one_to_one").dropna(subset=OUTCOMES)
    weekly = aggregate_weeks(daily).merge(market, on="week", how="left", validate="many_to_one")
    four = aggregate_weeks(daily, 4)
    reg = regressions(weekly, four)
    rebound = compare_windows(daily, rates, metadata, reg)
    common = weekly.assign(use_market=weekly.valid_change & weekly[FACTORS].notna().all(axis=1))
    keys = ["currency", "tenor"]
    coverage = rebound[[*keys, "country", "ticker", "matched_2023_months", "matched_2026_months",
                        "matched_2023_cds", "matched_2026_cds", "matched_change_cds", "full_window"]]
    raw_stats = cds.groupby(keys).date.agg(cds_days="size", first_cds="min", last_cds="max")
    match_stats = daily.groupby(keys).date.agg(matched_days="size", first_match="min", last_match="max")
    week_stats = common.groupby(keys).agg(valid_weeks=("valid_level", "sum"),
                                            change_weeks=("valid_change", "sum"),
                                            market_change_weeks=("use_market", "sum"))
    coverage = coverage.join(raw_stats, on=keys).join(match_stats, on=keys).join(week_stats, on=keys)
    base = reg.query("spec == 'market_baseline' and factor == 'dlog_vix' and hac_weeks == 4")
    plus = reg.query("spec == 'market_plus_CDS' and factor == 'd_cds' and hac_weeks == 4")
    market_fit = base.merge(plus, on=["tenor", "outcome", "n", "weeks", "countries", "first_week", "last_week"],
                           suffixes=("_base", "_plus"), validate="one_to_one")
    assert len(market_fit) == 6
    market_fit["delta_within_r2"] = market_fit.within_r2_plus - market_fit.within_r2_base
    market_fit["partial_r2_cds"] = 1 - market_fit.rss_plus / market_fit.rss_base
    relative, us_matched = relative_regressions(daily, market, load_us(us_path))
    rel_country, rel_summary = relative_rebound(rebound, relative, us_matched)
    for name, frame in [("relative_regressions", relative), ("relative_country_rebound", rel_country),
                        ("relative_rebound_summary", rel_summary), ("coverage", coverage), ("regressions", reg), ("market_incremental_fit", market_fit),
                        ("country_rebound", rebound), ("rebound_summary", attribution_summary(rebound))]:
        frame.to_csv(output / f"sovereign_cds_{name}.csv", index=False, float_format="%.9g")
    plot_rebound(rebound, output / "sovereign_cds_country_rebound.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--us-cds", type=Path, help="Daily US quotes with date, tenor and us_cds bps")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(output=args.output, us_path=args.us_cds)
