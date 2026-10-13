"""Fixed-cohort Treasury convenience and break-safe supply/inventory associations."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

CLEAN = ROOT / "output" / "cleaned"
TABLE = ROOT / "output" / "tables"
FIGURE = ROOT / "output" / "figures"
REPORT = ROOT / "output" / "report"
OFFICIAL = ROOT / "data" / "official"
# Selected for long historical coverage in both tenors, not regression results.
COHORT = ["AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "NZD", "SEK"]
METRICS = ["x_govt_raw_bps", "diff_y_raw_bps", "rho_raw_bps", "x_rf_raw_bps",
           "us_swap_spread_bps", "foreign_swap_spread_bps"]
CONTROLS = ["move_index", "vix_index", "bbdxy_index",
            "us_generic_govt_5y_yield", "ust_10y_2y_slope"]
BUCKETS = {"5y": (3, 6, "PDPOSGSC-G3L6"), "10y": (7, 11, "PDPOSGSC-G7L11")}


def same_interval_regime(labels: pd.Series, lag: int) -> pd.Series:
    """Require every month in the difference window to have one common regime."""
    valid = labels.notna()
    for offset in range(1, lag + 1):
        valid &= labels.eq(labels.shift(offset)) & labels.shift(offset).notna()
    return valid


def calendar_hac(model, dates: pd.DatetimeIndex, maxlags: int) -> np.ndarray:
    """Bartlett Newey-West covariance with real calendar gaps, not row-order lags."""
    months = dates.to_period("M").asi8
    assert len(np.unique(months)) == len(months), "One regression observation per month"
    scores = np.zeros((int(months.max() - months.min() + 1), model.model.exog.shape[1]))
    scores[months - months.min()] = model.model.exog * np.asarray(model.resid)[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(maxlags, len(scores) - 1) + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (maxlags + 1)) * (cross + cross.T)
    bread = np.asarray(model.normalized_cov_params)
    n, k = model.model.exog.shape
    return np.asarray(bread @ meat @ bread) * n / (n - k)


def build_convenience() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    raw = pd.read_csv(CLEAN / "raw_g10_benchmark_eq6_5y10y_panel.csv",
                      parse_dates=["date"], low_memory=False)
    raw = raw[raw["is_selected_regime_by_public_break"]].copy()
    assert not raw.duplicated(["date", "currency", "tenor"]).any()
    raw["diff_y_raw_bps"] = 100 * (raw["foreign_govt_yield"] - raw["usd_govt_yield"])
    # June 2026 has only three days: never treat it as a complete monthly observation.
    cutoff = raw["date"].max().to_period("M").start_time - pd.Timedelta(days=1)
    raw = raw[raw["date"].le(cutoff)]
    raw["month"] = raw["date"].dt.to_period("M").dt.to_timestamp("M")
    grouped = raw.groupby(["month", "currency", "tenor"])
    country = grouped[METRICS].mean().join(grouped.size().rename("n_days"))
    country = country.join(grouped["regime"].nunique().rename("n_regimes"))
    country = country.join(grouped["regime"].first().rename("regime")).reset_index()
    country.loc[country["n_days"].lt(15), METRICS] = np.nan
    country.to_csv(CLEAN / "treasury_interim_country_monthly.csv", index=False)
    months = pd.date_range(country["month"].min(), cutoff, freq="ME")
    panel = pd.DataFrame(index=months)
    coverage = []
    for tenor in BUCKETS:
        sub = country[country["tenor"].eq(tenor)]
        for ccy, group in sub.groupby("currency"):
            good = group.dropna(subset=["x_govt_raw_bps"])
            coverage.append({"currency": ccy, "tenor": tenor, "n_valid_months": len(good),
                             "first_month": good["month"].min(), "last_month": good["month"].max(),
                             "in_fixed8": ccy in COHORT})
        wide = sub.pivot(index="month", columns="currency", values="x_govt_raw_bps")
        panel[f"available_median_{tenor}"] = wide.median(axis=1)
        panel[f"available_count_{tenor}"] = wide.count(axis=1)
        for label, ccys in [("fixed8", COHORT), ("fixed7", [c for c in COHORT if c != "DKK"])]:
            complete = wide.reindex(columns=ccys).notna().all(axis=1)
            panel[f"{label}_mean_{tenor}"] = wide[ccys].mean(axis=1).where(complete)
            panel[f"{label}_median_{tenor}"] = wide[ccys].median(axis=1).where(complete)
        for metric in METRICS[1:]:
            values = sub.pivot(index="month", columns="currency", values=metric)[COHORT]
            panel[f"fixed8_{metric}_{tenor}"] = values.mean(axis=1).where(values.notna().all(axis=1))
        regimes = sub.pivot(index="month", columns="currency", values="regime").reindex(columns=COHORT)
        counts = sub.pivot(index="month", columns="currency", values="n_regimes").reindex(columns=COHORT)
        uniform = regimes.notna().all(axis=1) & counts.eq(1).all(axis=1)
        signature = regimes.fillna("missing").agg("|".join, axis=1).where(uniform)
        panel[f"benchmark_regime_{tenor}"] = signature
    panel.index.name = "date"
    return panel, pd.DataFrame(coverage), cutoff


def build_nominal_supply() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(OFFICIAL / "fiscal_mspd_table_3_market.csv", low_memory=False)
    raw["date"] = pd.to_datetime(raw["record_date"])
    raw["outstanding_amt"] = pd.to_numeric(raw["outstanding_amt"], errors="coerce")
    nominal = raw[raw["security_class1_desc"].isin(["Notes", "Bonds"])].copy()
    # Class totals have no maturity; issuance/reopening lines with the same CUSIP can be additive.
    securities = nominal[nominal["security_class2_desc"].str.fullmatch(r"[0-9A-Z]{9}", na=False)].copy()
    securities["maturity"] = pd.to_datetime(securities["maturity_date"], errors="coerce")
    securities = securities.dropna(subset=["maturity", "outstanding_amt"])
    assert not securities.duplicated(["date", "src_line_nbr"]).any()
    securities = securities[securities["maturity"].gt(securities["date"])]
    totals = securities.groupby(["date", "security_class1_desc"])["outstanding_amt"].sum()
    reference = nominal[nominal["security_class2_desc"].str.startswith("Total Unmatured", na=False)]
    reference = reference.set_index(["date", "security_class1_desc"])["outstanding_amt"]
    audit = pd.concat([totals.rename("security_sum_mil"), reference.rename("reported_total_mil")], axis=1)
    audit["difference_mil"] = audit["security_sum_mil"] - audit["reported_total_mil"]
    audit["relative_error"] = audit["difference_mil"].abs() / audit["reported_total_mil"]
    assert audit.notna().all().all(), "Reconciliation must cover every class/month"
    assert audit["relative_error"].max() < 1e-6, "MSPD security totals must reconcile"
    supply = totals.groupby("date").sum().to_frame("nominal_coupon_total_mil")
    years = (securities["maturity"] - securities["date"]).dt.days / 365.25
    for tenor, (lower, upper, _) in BUCKETS.items():
        bucket = securities[years.gt(lower) & years.le(upper)].groupby("date")["outstanding_amt"].sum()
        supply[f"supply_trillion_{tenor}"] = bucket / 1e6
        supply[f"supply_share_pct_{tenor}"] = 100 * bucket / supply["nominal_coupon_total_mil"]
    return supply, audit.reset_index()


def build_dealer() -> pd.DataFrame:
    definitions = json.loads((ROOT / "data" / "source_discovery" / "nyfed_pd_seriesbreaks.json").read_text())["pd"]["seriesbreaks"]
    parts = []
    for tenor, (_, _, key) in BUCKETS.items():
        weekly = pd.read_csv(OFFICIAL / f"nyfed_{key}.csv", parse_dates=["date"])
        weekly["regime"] = pd.Series(index=weekly.index, dtype="object")
        for definition in definitions:
            stop = min(definition["enddate"], "2099-12-31")
            mask = weekly["date"].between(definition["startdate"], stop)
            weekly.loc[mask, "regime"] = definition["seriesbreak"]
        assert weekly["regime"].notna().all()
        weekly["month"] = weekly["date"].dt.to_period("M").dt.to_timestamp("M")
        groups = weekly.groupby("month")
        monthly = groups["value"].mean().div(1000).to_frame(f"dealer_billion_{tenor}")
        valid = groups["regime"].nunique().eq(1) & groups["value"].count().ge(3)
        monthly[f"dealer_regime_{tenor}"] = groups["regime"].first().where(valid)
        monthly[f"dealer_billion_{tenor}"] = monthly[f"dealer_billion_{tenor}"].where(valid)
        parts.append(monthly)
    return pd.concat(parts, axis=1)


def model_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefs, audits = [], []
    for tenor in BUCKETS:
        for cohort in ["fixed8_mean", "fixed8_median", "fixed7_mean"]:
            for transform, lag, maxlags in [("level_trend", 0, 12), ("change_1m", 1, 6), ("change_12m", 12, 12)]:
                for supply_type, episode in [("trillion", "all"), ("share_pct", "all"), ("trillion", "exclude_covid_window")]:
                    # Composition is a pre-specified robustness check, not an alternate main outcome.
                    if supply_type == "share_pct" and (cohort != "fixed8_mean" or lag != 12):
                        continue
                    if episode != "all" and (cohort != "fixed8_mean" or lag != 12):
                        continue
                    ycol = f"{cohort}_{tenor}"
                    predictors = [f"supply_{supply_type}_{tenor}", f"dealer_billion_{tenor}", *CONTROLS]
                    sample = panel.loc["2013-04-30":, [ycol, *predictors]].copy()
                    valid = panel[f"dealer_regime_{tenor}"].notna() & panel[f"benchmark_regime_{tenor}"].notna()
                    if lag:
                        sample = panel[[ycol, *predictors]].diff(lag).loc[sample.index]
                        valid &= same_interval_regime(panel[f"dealer_regime_{tenor}"], lag)
                        valid &= same_interval_regime(panel[f"benchmark_regime_{tenor}"], lag)
                    if episode != "all":
                        # Drop whole annual-difference windows touching the March-June 2020 shock.
                        start = panel.index - pd.DateOffset(months=lag)
                        valid &= ~((panel.index >= pd.Timestamp("2020-03-01"))
                                   & (start <= pd.Timestamp("2020-06-30")))
                    sample = sample.where(valid.loc[sample.index], np.nan).dropna()
                    assert len(sample) >= 48
                    scales = sample[predictors].std()
                    x = (sample[predictors] - sample[predictors].mean()) / scales
                    if not lag:
                        x["trend_years"] = (sample.index - sample.index.min()).days / 365.25
                        joint_regime = (panel.loc[sample.index, f"dealer_regime_{tenor}"] + "|"
                                        + panel.loc[sample.index, f"benchmark_regime_{tenor}"])
                        x = x.join(pd.get_dummies(joint_regime, prefix="joint_regime", drop_first=True, dtype=float))
                    model = sm.OLS(sample[ycol], sm.add_constant(x)).fit()
                    assert np.linalg.matrix_rank(model.model.exog) == model.model.exog.shape[1]
                    bands = [maxlags, 18] if lag == 12 else [maxlags]
                    for bandwidth in bands:
                        cov = calendar_hac(model, sample.index, bandwidth)
                        se = np.sqrt(np.maximum(np.diag(cov), 0))
                        for variable in predictors[:2]:
                            idx = model.params.index.get_loc(variable)
                            beta, stderr = model.params.iloc[idx], se[idx]
                            coefs.append({"tenor": tenor, "cohort": cohort, "transform": transform,
                                          "supply_type": supply_type, "episode": episode, "variable": variable,
                                          "coef_bps_per_sd": beta, "se_calendar_hac": stderr,
                                          "ci95_low": beta - 1.96 * stderr, "ci95_high": beta + 1.96 * stderr,
                                          "p_value": 2 * norm.sf(abs(beta / stderr)), "hac_months": bandwidth,
                                          "raw_sd": scales[variable], "coef_bps_per_raw_unit": beta / scales[variable],
                                          "n_months": len(sample), "r_squared": model.rsquared,
                                          "first_month": sample.index.min(), "last_month": sample.index.max()})
                    audits.append({"tenor": tenor, "cohort": cohort, "transform": transform,
                                   "supply_type": supply_type, "episode": episode, "n_complete_months": len(sample),
                                   "n_excluded_regime_windows": int((~valid.loc[panel.loc["2013-04-30":].index]).sum()),
                                   "calendar_span_months": int(np.ptp(sample.index.to_period("M").asi8)) + 1,
                                   "n_design_columns": model.model.exog.shape[1]})
    return pd.DataFrame(coefs), pd.DataFrame(audits)


def result_summaries(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = {"2000-2006": ("2000", "2006"), "2010-2020": ("2010", "2020"), "2021-2025": ("2021", "2025"),
               "2023-calendar-year": ("2023", "2023"), "2026-Jan-May": ("2026-01", "2026-05")}
    rows, bridges = [], []
    for tenor in BUCKETS:
        for metric in ["fixed8_mean", "fixed8_median", "fixed7_mean", "available_median"]:
            for label, (start, stop) in periods.items():
                values = panel.loc[start:stop, f"{metric}_{tenor}"].dropna()
                rows.append({"tenor": tenor, "aggregation": metric, "period": label,
                             "mean_bps": values.mean(), "n_months": len(values)})
        columns = [f"fixed8_mean_{tenor}", *[f"fixed8_{m}_{tenor}" for m in METRICS[1:]]]
        for label, start, stop in [("2023 average to 2026 Jan-May average", "2023", "2026-01:2026-05"),
                                   ("2023 trough month to latest complete month", "trough", "latest")]:
            if start == "trough":
                first = panel.loc["2023", columns[0]].idxmin()
                last = panel[columns[0]].last_valid_index()
                delta = panel.loc[last, columns] - panel.loc[first, columns]
                first_label, last_label = str(first.date()), str(last.date())
            else:
                first_values = panel.loc[start, columns].dropna()
                last_values = panel.loc["2026-01":"2026-05", columns].dropna()
                delta = last_values.mean() - first_values.mean()
                first_label, last_label = "2023 annual average", "2026 Jan-May average"
            bridges.append({"tenor": tenor, "comparison": label, "start": first_label, "end": last_label,
                            "convenience_change_bps": delta.iloc[0], "diff_y_change_bps": delta.iloc[1],
                            "rho_change_bps": delta.iloc[2], "dollar_proxy_change_bps": delta.iloc[3],
                            "us_swap_spread_change_bps": delta.iloc[4], "foreign_swap_spread_change_bps": delta.iloc[5],
                            "eq5_bridge_error_bps": delta.iloc[0] - delta.iloc[1] + delta.iloc[2],
                            "eq6_bridge_error_bps": delta.iloc[0] - delta.iloc[3] - delta.iloc[4] + delta.iloc[5]})
    return pd.DataFrame(rows), pd.DataFrame(bridges)


def component_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = []
    for tenor in BUCKETS:
        for label, start, stop in [("2000-2006", "2000", "2006"), ("2010-2020", "2010", "2020"), ("2021-2025", "2021", "2025"),
                                   ("2023", "2023", "2023"), ("2026-Jan-May", "2026-01", "2026-05")]:
            cols = [f"fixed8_mean_{tenor}", *[f"fixed8_{m}_{tenor}" for m in METRICS[1:]]]
            sample = panel.loc[start:stop, cols].dropna()
            row = {"tenor": tenor, "period": label, "n_months": len(sample)}
            row.update(dict(zip(METRICS, sample.mean())))
            components.append(row)
    country = pd.read_csv(CLEAN / "treasury_interim_country_monthly.csv", parse_dates=["month"])
    country = country[country["currency"].isin(COHORT)]
    baseline = country[country["month"].dt.year.eq(2023)].groupby(["currency", "tenor"])[METRICS].mean()
    endpoint = country[country["month"].between("2026-01-01", "2026-05-31")].groupby(["currency", "tenor"])[METRICS].mean()
    changes = (endpoint - baseline).add_suffix("_change_bps").reset_index()
    return pd.DataFrame(components), changes


def make_figures(panel: pd.DataFrame, coefs: pd.DataFrame) -> None:
    colors = {"5y": "#246b91", "10y": "#ab6c18"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
    for tenor, ax in zip(BUCKETS, axes):
        for name, style, color in [("fixed8_mean", "-", colors[tenor]), ("fixed8_median", "--", "#454545"),
                                   ("available_median", ":", "#a5576c")]:
            ax.plot(panel.index, panel[f"{name}_{tenor}"], label=name.replace("_", " "), ls=style, color=color, lw=1.3)
        ax.axhline(0, color="#444444", lw=0.7)
        ax.set_ylabel(f"{tenor.upper()} convenience (bps)")
        ax.legend(frameon=False, ncol=3, fontsize=9)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_title("Treasury convenience: fixed composition versus changing coverage")
    axes[-1].set_xlim(panel.index.min(), panel.index.max())
    axes[-1].set_xticks([panel.index[0], *pd.to_datetime(["2005-01-31", "2010-01-31", "2015-01-31", "2020-01-31"]), panel.index[-1]],
                       labels=["2000-01", "2005", "2010", "2015", "2020", "2026-05"])
    fig.tight_layout()
    fig.savefig(FIGURE / "treasury_interim_fixed_cohort.png", dpi=170)
    plt.close(fig)
    primary = coefs[(coefs["cohort"].eq("fixed8_mean")) & coefs["supply_type"].eq("trillion") & coefs["episode"].eq("all")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for tenor, ax in zip(BUCKETS, axes):
        sub = primary[primary["tenor"].eq(tenor)].copy()
        labels = [f"{r.transform}; HAC {r.hac_months}\n{'Supply' if r.variable.startswith('supply') else 'Net inventory'}"
                  for r in sub.itertuples()]
        y = np.arange(len(sub))
        ax.errorbar(sub["coef_bps_per_sd"], y, xerr=1.96 * sub["se_calendar_hac"], fmt="o", color=colors[tenor], capsize=3)
        ax.set_yticks(y, labels=labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="#444444", lw=0.8)
        ax.set_title(tenor.upper())
        ax.set_xlabel("bps per sample SD; 95% calendar-HAC interval")
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Supply / dealer inventory associations, controlled; not causal estimates")
    fig.tight_layout()
    fig.savefig(FIGURE / "treasury_interim_associations.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
    for tenor, ax in zip(BUCKETS, axes):
        sub = panel.loc["2023":]
        series = {"Treasury convenience": sub[f"fixed8_mean_{tenor}"],
                  "Dollar proxy": sub[f"fixed8_x_rf_raw_bps_{tenor}"],
                  "U.S. swap spread": sub[f"fixed8_us_swap_spread_bps_{tenor}"],
                  "Minus foreign swap spread": -sub[f"fixed8_foreign_swap_spread_bps_{tenor}"]}
        for (label, values), color, style in zip(series.items(), ["#222222", "#246b91", "#ab6c18", "#a5576c"], ["-", "--", ":", "-."]):
            ax.plot(values.index, values - values.iloc[0], label=label, color=color, ls=style, lw=1.5)
        ax.set_ylabel(f"{tenor.upper()} change (bps)")
        ax.axhline(0, color="#444444", lw=0.7)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, ncol=2, fontsize=9)
    axes[0].set_title("Additive Eq.6 accounting: fixed-eight means, rebased to January 2023")
    axes[-1].set_xlim(sub.index.min(), sub.index.max())
    axes[-1].set_xticks(pd.to_datetime(["2023-01-31", "2024-01-31", "2025-01-31", "2026-01-31", "2026-05-31"]),
                       labels=["2023-01", "2024-01", "2025-01", "2026-01", "2026-05"])
    fig.tight_layout()
    fig.savefig(FIGURE / "treasury_interim_rebound_components.png", dpi=170)
    plt.close(fig)


def write_memo(periods: pd.DataFrame, bridges: pd.DataFrame, coefs: pd.DataFrame, audit: pd.DataFrame, cutoff) -> None:
    primary = coefs[coefs["cohort"].eq("fixed8_mean") & coefs["supply_type"].eq("trillion") & coefs["episode"].eq("all")]
    findings = []
    for tenor in BUCKETS:
        early = periods[periods.tenor.eq(tenor) & periods.aggregation.eq("fixed8_mean") & periods.period.eq("2000-2006")].iloc[0]
        late = periods[periods.tenor.eq(tenor) & periods.aggregation.eq("fixed8_mean") & periods.period.eq("2021-2025")].iloc[0]
        rebound = bridges[bridges.tenor.eq(tenor)].iloc[0]
        findings.append(f"- {tenor.upper()}: available complete-month 2000-2006 mean {early.mean_bps:.2f} bps ({early.n_months} months), versus 2021-2025 {late.mean_bps:.2f} bps ({late.n_months} months): change {late.mean_bps - early.mean_bps:.2f} bps. Early coverage is incomplete, so this is not a full balanced 2000-2006 paper sample.")
        findings.append(f"- {tenor.upper()} recent recovery: {rebound.convenience_change_bps:+.2f} bps comparing 2023 average with 2026 Jan-May average. Eq.6 contributions: dollar proxy {rebound.dollar_proxy_change_bps:+.2f}, U.S. swap spread {rebound.us_swap_spread_change_bps:+.2f}, minus foreign swap spread {-rebound.foreign_swap_spread_change_bps:+.2f} bps. This does not establish that U.S. absolute safe-asset quality improved.")
    lines = ["# Treasury Convenience: Interim Research Results", "", "Reviewed: 2026-10-02", "",
             f"Complete-month endpoint: {cutoff.date()}. Raw Bloomberg quotes end on 2026-06-03; no later market data were refreshed.", "",
             "## Executive Findings", "", *findings,
             "- Absolute supply changes have negative controlled annual-change associations; one-month effects are weak. Supply-share robustness is materially weaker. Do not interpret the absolute-supply results as proof of maturity-specific scarcity.",
             "- The 10Y net-inventory association is negative in the main annual-change model, but depends on specification. The 5Y inventory association is not supported. These data do not establish a general dealer-capacity mechanism.", "",
             "## Research Design", "",
             "Part A reconstructs government-bond CIP using raw yields and swap-implied hedging premia (Eq.5), with the public V4 series as a benchmark through June 2025. It is not an independent outright-forward estimator at 5Y/10Y.",
             "Part B studies U.S. nominal maturity-bucket supply and primary-dealer net inventory. This is an exploratory extension, not a replication of the paper's country/GDP regressions or a causal capacity estimate.",
             "Fixed eight-currency cohort: " + ", ".join(COHORT) + ". JPY and NOK lack full early histories. Each currency/month requires 15 quotes. Equal-weight country monthly means preserve the decomposition; medians are a robustness check, not an additive decomposition.",
             "Long-history comparisons span IBOR/OIS benchmark definitions; they describe the authors' stitched measure, not a constant-benchmark causal change. The early fixed-cohort window has only 48/49 complete months, not all 84 months. DKK exclusion, medians, and the pre-transition 2010-2020 window are reported to expose this sensitivity.",
             "Nominal Notes/Bonds only; MSPD issue/reopening line amounts are summed without class totals. Inventory buckets: (3,6]Y and (7,11]Y. The latter differs from the paper's 6-11Y supply bucket.",
             f"Security sums reconcile against reported unmatured class totals: maximum relative discrepancy {audit.relative_error.max():.2g}.",
             "Controls: MOVE, VIX, broad dollar, U.S. 5Y yield and 10Y-2Y slope. Post-April-2013 models are fixed in advance: trend-adjusted levels, 1-month changes, 12-month changes; supply-share and cohort sensitivity. Entire difference windows crossing dealer reporting regimes or currency benchmark conventions are excluded. Calendar-aware Bartlett HAC retains gaps; 12-month changes use 12/18 lags.", "",
             "## Historical Levels", "", periods.round(3).to_markdown(index=False), "",
             "## Recent Rebound: Additive Accounting, Not Causal Attribution", "", bridges.round(3).to_markdown(index=False), "",
             "Trough-to-end deltas are ex-post, selected-endpoint descriptions, not event-study estimates. Calendar-year-to-year-average deltas provide a less selected comparison.", "",
             "All eight cohort currencies show a decline in foreign swap spreads in the 2023-to-2026-average comparison, while the 5Y Treasury-convenience rebound is positive for seven/eight (DKK is the exception). This is broad relative-spread accounting, not one currency alone driving the average.", "",
             "## Main Association Estimates", "",
             primary[["tenor", "transform", "variable", "hac_months", "coef_bps_per_sd", "ci95_low", "ci95_high", "p_value", "n_months"]].round(3).to_markdown(index=False), "",
             "Full coefficient tables retain units, sample scales, composition sensitivity and uncertainty. No specification is selected because it is significant; p-values are unadjusted exploratory diagnostics, not confirmatory tests.", "",
             "Excluding complete annual-difference windows touching March-June 2020 retains negative absolute-supply estimates, but the 10Y inventory interval then includes zero. Supply-share intervals include zero at both tenors. Thus the supply association is the stronger descriptive lead; a general dealer-capacity claim is premature.", "",
             "## Pre-Specified Sensitivity Results", "",
             coefs[coefs["transform"].eq("change_12m") & coefs.hac_months.eq(12)][["tenor", "cohort", "supply_type", "episode", "variable", "coef_bps_per_sd", "ci95_low", "ci95_high", "n_months"]].round(3).to_markdown(index=False), "",
             "## What These Results Do Not Establish", "",
             "An Eq.6 identity residual near zero verifies bookkeeping only, not an independent dollar-convenience estimate. Treasury convenience is relative to foreign governments, not a standalone absolute U.S. premium. U.S. supply and dealer inventory are endogenous; net positions measure exposure, not spare balance-sheet capacity. The foreign supply/GDP, Fed holdings, and causal identification needed for a stronger interpretation remain outside this bounded extension.", "",
             "## Next Decision", "",
             "Use the fixed-cohort, break-safe results as the interim presentation. Keep the validated raw reconstruction as the project's strongest completed contribution. Before a final causal claim, agree on one identification design and its incremental data with Samim. A final descriptive replication-plus-robustness report does not require completing EM/3M or all figures in the paper.", "",
             "## Sources", "",
             "- Public V4 data and ticker conventions: https://sites.google.com/view/jschreger/CIP",
             "- MSPD: https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/",
             "- NY Fed positions and reporting breaks: https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics",
             "- Bloomberg inputs: six retained Excel downloads; no redistribution of licensed raw inputs."]
    (REPORT / "Treasury_Convenience_Interim_Results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for folder in [CLEAN, TABLE, FIGURE, REPORT]:
        folder.mkdir(parents=True, exist_ok=True)
    panel, coverage, cutoff = build_convenience()
    supply, supply_audit = build_nominal_supply()
    dealer = build_dealer()
    controls = pd.read_csv(CLEAN / "eq5_ust_extension_monthly_panel.csv", parse_dates=["date"]).set_index("date")[CONTROLS]
    panel = panel.join(supply).join(dealer).join(controls)
    coefs, model_audit = model_tables(panel)
    periods, bridges = result_summaries(panel)
    components, country_changes = component_tables(panel)
    assert bridges[["eq5_bridge_error_bps", "eq6_bridge_error_bps"]].abs().max().max() < 1e-8
    panel.to_csv(CLEAN / "treasury_interim_monthly_panel.csv")
    for frame, name in [(coverage, "cohort_coverage"), (supply_audit, "supply_reconciliation"),
                        (model_audit, "model_audit"), (coefs, "coefficients"),
                        (periods, "period_means"), (bridges, "rebound_decomposition"),
                        (components, "component_periods"), (country_changes, "country_rebound")]:
        frame.to_csv(TABLE / f"treasury_interim_{name}.csv", index=False)
    make_figures(panel, coefs)
    write_memo(periods, bridges, coefs, supply_audit, cutoff)
    print("Complete-month cutoff:", cutoff.date())
    print(periods[periods.aggregation.eq("fixed8_mean")].to_string(index=False))
    print(coefs[coefs.cohort.eq("fixed8_mean") & coefs.supply_type.eq("trillion")].to_string(index=False))


if __name__ == "__main__":
    main()
