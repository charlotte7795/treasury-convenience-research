"""Reviewed final descriptive study, with market quotes refreshed through October 2026."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analyze_foreign_swap_spread_rebound import METRICS as LEG_METRICS, country_monthly
from src.analyze_treasury_interim import CLEAN, COHORT, FIGURE, REPORT, TABLE, model_tables, plt
from src.refresh_treasury_5y10y import PANEL

COMPONENTS = [*LEG_METRICS, "diff_y_raw_bps", "rho_raw_bps"]
BASELINES = {"2023 annual average": ("2023-01-31", "2023-12-31"),
             "2023 H2 average": ("2023-07-31", "2023-12-31"),
             "2024 annual average": ("2024-01-31", "2024-12-31")}
ENDPOINTS = {"2026 Jan-May average": ("2026-01-31", "2026-05-31"),
             "2026 Jan-Sep average": ("2026-01-31", "2026-09-30"),
             "2026 Q3 average": ("2026-07-31", "2026-09-30"),
             "September 2026": ("2026-09-30", "2026-09-30")}
SOURCES = {
    "paper": "https://www.federalreserve.gov/econres/ifdp/decoupling-dollar-and-treasury-privilege.htm",
    "public": "https://sites.google.com/view/jschreger/CIP",
    "bis": "https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption",
    "ecb": "https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html",
    "supply": "https://fiscaldata.treasury.gov/datasets/monthly-statement-public-debt/",
    "dealer": "https://www.newyorkfed.org/markets/counterparties/primary-dealers-statistics",
}


def complete_country_months(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    selected = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT)
                   & raw.date.ge("2023-01-01")].copy()
    cutoff = selected.date.max().to_period("M").start_time - pd.Timedelta(days=1)
    selected = selected[selected.date.le(cutoff)]
    monthly = country_monthly(selected)
    monthly["diff_y_raw_bps"] = monthly.foreign_govt_yield_bps - monthly.usd_govt_yield_bps
    monthly["rho_raw_bps"] = (monthly.foreign_swap_bps - monthly.usd_swap_bps
                              - monthly.x_rf_raw_bps)
    months = pd.date_range("2023-01-31", cutoff, freq="ME")
    assert set(monthly.month) == set(months)
    assert monthly.groupby(["currency", "tenor"]).size().eq(len(months)).all()
    assert monthly[COMPONENTS].notna().all().all(), "Fixed-cohort month missing or below 15 quotes"
    assert len(monthly) == len(COHORT) * 2 * len(months)
    assert np.allclose(monthly.x_govt_raw_bps, monthly.diff_y_raw_bps - monthly.rho_raw_bps, atol=1e-8)
    assert np.allclose(monthly.x_govt_raw_bps,
                       monthly.x_rf_raw_bps + monthly.us_swap_spread_bps - monthly.foreign_swap_spread_bps,
                       atol=1e-8)
    return monthly, cutoff


def period_changes(monthly: pd.DataFrame, before: tuple[str, str], after: tuple[str, str]) -> pd.DataFrame:
    keys = ["currency", "tenor"]
    start = monthly[monthly.month.between(*before)].groupby(keys)[COMPONENTS].mean()
    stop = monthly[monthly.month.between(*after)].groupby(keys)[COMPONENTS].mean()
    assert len(start) == len(stop) == 16
    delta = (stop - start).reset_index()
    delta["eq5_bridge_error_bps"] = delta.x_govt_raw_bps - delta.diff_y_raw_bps + delta.rho_raw_bps
    delta["eq6_bridge_error_bps"] = (delta.x_govt_raw_bps - delta.x_rf_raw_bps
                                     - delta.us_swap_spread_bps + delta.foreign_swap_spread_bps)
    delta["without_foreign_spread_change_bps"] = delta.x_rf_raw_bps + delta.us_swap_spread_bps
    assert delta[["eq5_bridge_error_bps", "eq6_bridge_error_bps"]].abs().max().max() < 1e-8
    return delta


def sensitivity_tables(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    changes, summaries = [], []
    for baseline, before in BASELINES.items():
        for endpoint, after in ENDPOINTS.items():
            delta = period_changes(monthly, before, after)
            delta["baseline"], delta["endpoint"] = baseline, endpoint
            changes.append(delta)
            exclusions = ["none", *COHORT] if baseline == "2023 annual average" else ["none"]
            for excluded in exclusions:
                for tenor, sub in delta[delta.currency.ne(excluded)].groupby("tenor"):
                    summaries.append({"baseline": baseline, "endpoint": endpoint,
                                      "excluded_currency": excluded, "tenor": tenor,
                                      "n_currencies": len(sub), "positive_convenience_changes": int(sub.x_govt_raw_bps.gt(0).sum()),
                                      "negative_foreign_spread_changes": int(sub.foreign_swap_spread_bps.lt(0).sum()),
                                      **sub[COMPONENTS + ["without_foreign_spread_change_bps"]].mean().to_dict()})
    return pd.concat(changes, ignore_index=True), pd.DataFrame(summaries)


def paired_validation_trend(raw: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    selected = raw[raw.is_selected_regime_by_public_break & raw.tenor.eq("5y")]
    paired = selected[["date", "currency", "tenor", "x_govt_raw_bps"]].merge(
        public[["date", "currency", "tenor", "cip_govt"]],
        on=["date", "currency", "tenor"], validate="one_to_one").dropna()
    daily = paired.groupby("date")[["x_govt_raw_bps", "cip_govt"]].median()
    monthly = daily.resample("ME").mean().dropna()
    monthly.columns = ["Raw Bloomberg", "Public V4 benchmark"]
    return monthly


def make_figures(aggregate: pd.DataFrame, primary: pd.DataFrame, changes: pd.DataFrame,
                 validation_trend: pd.DataFrame) -> None:
    colors = {"5y": "#246b91", "10y": "#a65b3e"}
    fig, ax = plt.subplots(figsize=(10.5, 5))
    for tenor in colors:
        sub = aggregate[aggregate.tenor.eq(tenor)].set_index("month")
        ax.plot(sub.index, sub.x_govt_raw_bps, lw=2, color=colors[tenor], label=tenor.upper())
    ax.axhline(0, color="#454545", lw=.8)
    ax.set_ylabel("Treasury convenience (bps)")
    ax.set_title("Fixed-eight Treasury convenience, monthly means")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_xlim(aggregate.month.min(), aggregate.month.max())
    ax.set_xticks(pd.to_datetime(["2023-01-31", "2024-01-31", "2025-01-31", "2026-01-31", "2026-09-30"]),
                  ["2023-01", "2024-01", "2025-01", "2026-01", "2026-09"])
    fig.tight_layout()
    fig.savefig(FIGURE / "treasury_final_recent_convenience.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = ["Dollar\nproxy", "U.S. swap\nspread", "Minus foreign\nswap spread", "Net\nconvenience"]
    for tenor, ax in zip(colors, axes):
        row = primary[primary.tenor.eq(tenor)].iloc[0]
        values = [row.x_rf_raw_bps, row.us_swap_spread_bps, -row.foreign_swap_spread_bps, row.x_govt_raw_bps]
        ax.bar(labels, values, color=["#246b91", "#a65b3e", "#426b50", "#343434"])
        for i, value in enumerate(values):
            ax.annotate(f"{value:+.1f}", (i, value), xytext=(0, 5 if value >= 0 else -15), textcoords="offset points", ha="center")
        ax.axhline(0, color="#454545", lw=.8)
        ax.set_ylim(-20, 45)
        ax.set_title(tenor.upper())
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Change / accounting contribution (bps)")
    fig.suptitle("2023 average to 2026 Jan-Sep average: Eq.6 accounting")
    fig.tight_layout()
    fig.savefig(FIGURE / "treasury_final_rebound_accounting.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for tenor, ax in zip(colors, axes):
        sub = changes[changes.tenor.eq(tenor)].set_index("currency").loc[COHORT]
        ax.bar(np.arange(8)-.18, sub.foreign_swap_bps, .36, color="#246b91", label="Swap rate")
        ax.bar(np.arange(8)+.18, -sub.foreign_govt_yield_bps, .36, color="#a65b3e", label="Minus government yield")
        ax.scatter(np.arange(8), sub.foreign_swap_spread_bps, color="#222222", label="Net spread", zorder=3)
        ax.axhline(0, color="#454545", lw=.8)
        ax.set_xticks(np.arange(8), COHORT)
        ax.set_title(tenor.upper())
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Change / accounting contribution (bps)")
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle("Foreign swap spreads: country-level raw-leg changes")
    fig.tight_layout()
    fig.savefig(FIGURE / "treasury_final_foreign_legs.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.plot(validation_trend.index, validation_trend.iloc[:, 0], label="Raw Bloomberg", color="#246b91", lw=2)
    ax.plot(validation_trend.index, validation_trend.iloc[:, 1], label="Public V4", color="#a65b3e", lw=1.3, ls="--")
    ax.axhline(0, color="#454545", lw=.8)
    ax.set_ylabel("Government CIP (bps)")
    ax.set_title("5Y raw reconstruction versus public benchmark, paired observations")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(FIGURE / "treasury_final_validation_trend.png", dpi=180)
    plt.close(fig)


def main() -> dict:
    raw = pd.read_csv(PANEL, parse_dates=["date"], low_memory=False)
    monthly, cutoff = complete_country_months(raw)
    monthly.to_csv(CLEAN / "treasury_final_country_monthly.csv", index=False)
    aggregate = monthly.groupby(["month", "tenor"])[COMPONENTS].mean().reset_index()
    aggregate.to_csv(CLEAN / "treasury_final_monthly.csv", index=False)
    changes, sensitivity = sensitivity_tables(monthly)
    changes.to_csv(TABLE / "treasury_final_country_changes.csv", index=False)
    sensitivity.to_csv(TABLE / "treasury_final_rebound_sensitivity.csv", index=False)
    primary = sensitivity[sensitivity.baseline.eq("2023 annual average")
                          & sensitivity.endpoint.eq("2026 Jan-Sep average") & sensitivity.excluded_currency.eq("none")]
    countries = changes[changes.baseline.eq("2023 annual average") & changes.endpoint.eq("2026 Jan-Sep average")]
    model_panel = pd.read_csv(CLEAN / "treasury_interim_monthly_panel.csv", parse_dates=["date"]).set_index("date")
    # Controls and official supply/inventory are not refreshed: do not impute July-September predictors.
    for tenor, group in aggregate.groupby("tenor"):
        source = group.set_index("month")
        overlap = model_panel.index.intersection(source.index)
        model_panel.loc[overlap, f"fixed8_mean_{tenor}"] = source.loc[overlap, "x_govt_raw_bps"]
        for ccy_label, ccys in [("fixed8_median", COHORT), ("fixed7_mean", [c for c in COHORT if c != "DKK"])]:
            sub = monthly[monthly.tenor.eq(tenor) & monthly.currency.isin(ccys)].groupby("month").x_govt_raw_bps
            replacement = sub.median() if ccy_label == "fixed8_median" else sub.mean()
            model_panel.loc[overlap, f"{ccy_label}_{tenor}"] = replacement.loc[overlap]
    coefs, audit = model_tables(model_panel)
    coefs.to_csv(TABLE / "treasury_final_association_coefficients.csv", index=False)
    audit.to_csv(TABLE / "treasury_final_association_audit.csv", index=False)
    for column in ["first_month", "last_month"]:
        coefs[column] = pd.to_datetime(coefs[column]).dt.strftime("%Y-%m-%d")
    model_panel.to_csv(CLEAN / "treasury_final_association_panel.csv")
    public = pd.read_csv(CLEAN / "public_cip_cleaned.csv", parse_dates=["date"])
    public["tenor"] = public.tenor.str.lower()
    trend = paired_validation_trend(raw, public)
    trend.to_csv(TABLE / "treasury_final_paired_validation_trend.csv")
    make_figures(aggregate, primary, countries, trend)
    validation = pd.read_csv(TABLE / "treasury_final_replication_validation.csv")
    gov = validation[validation.check_type.eq("x_govt_vs_public_combined")]
    regime_gov = validation[validation.check_type.eq("x_govt_vs_public_regime")]
    fig3 = pd.read_csv(TABLE / "raw_figure3_long_tenors_vs_public_comparison.csv")
    main_models = coefs[coefs.cohort.eq("fixed8_mean") & coefs["transform"].eq("change_12m")
                        & coefs.supply_type.eq("trillion") & coefs.episode.eq("all") & coefs.hac_months.eq(12)]
    summary = {"updated": "2026-10-09", "complete_month_end": str(cutoff.date()), "cohort": COHORT,
               "country_tenor_months": len(monthly), "months_per_pair": int(monthly.groupby(["currency", "tenor"]).size().min()),
               "minimum_daily_quotes_per_month": int(monthly.n_days.min()),
               "gov_good_match": int(gov.status.eq("GOOD_MATCH").sum()), "gov_total_checks": len(gov),
               "gov_max_mae_bps": float(gov.mae_bps.max()), "gov_min_corr": float(gov["corr"].min()),
               "gov_regime_good_match": int(regime_gov.status.eq("GOOD_MATCH").sum()),
               "gov_regime_total_checks": len(regime_gov),
               "gov_regime_review": regime_gov[regime_gov.status.ne("GOOD_MATCH")].to_dict("records"),
               "figure3_good_match": int(fig3.status.eq("GOOD_MATCH").sum()), "figure3_total_checks": len(fig3),
               "association_end": str(pd.to_datetime(main_models.last_month).max().date()),
               "rebound": primary.to_dict("records"), "association_models": main_models.to_dict("records"),
               "sources": SOURCES,
               "scope": "Raw G10 5Y/10Y reconstruction plus a descriptive relative-convenience extension; not full-paper exact replication or causal identification."}
    (TABLE / "treasury_final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    payload = {"summary": summary, "monthly": aggregate.assign(month=aggregate.month.dt.strftime("%Y-%m")).to_dict("records"),
               "primary_country_changes": countries.to_dict("records"), "sensitivity": sensitivity.to_dict("records"),
               "gov_validation": gov.to_dict("records"),
               "validation_trend": trend.reset_index().assign(date=trend.index.strftime("%Y-%m")).to_dict("records"),
               "association_sensitivity": coefs[coefs["transform"].eq("change_12m") & coefs.hac_months.eq(12)].to_dict("records")}
    (TABLE / "treasury_final_presentation_data.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# Treasury Convenience: Final Research Findings", "", "Updated: 2026-10-09", "",
             f"Market accounting through {cutoff.date()}; supply/inventory regressions through {summary['association_end']}. These are different endpoints.", "",
             "## Scope", "", summary["scope"], "",
             "Equation 5 is estimated directly using government yields and the Appendix swap-implied hedging premium. Equation 6 is an additive explanation of that result, not an independent estimator. Dollar convenience is a benchmark/basis proxy, not an independently validated risk-free measure for every currency.", "",
             "The public Data Appendix says its data are not identical to those in the published papers. Thus a public-series match is not a claim of identical paper Table 2 observation counts or exact full-paper replication.", "",
             "Combined-regime government-CIP checks pass 20/20. Separate regime checks pass 39/40: DKK 5Y SOFR remains CHECK (corr 0.978, MAE 1.04 bps). This narrow diagnostic discrepancy is retained, not suppressed by the combined result.", "",
             "## Updated Recent Result", "", primary[["tenor", "x_govt_raw_bps", "x_rf_raw_bps", "us_swap_spread_bps", "foreign_swap_spread_bps", "diff_y_raw_bps", "rho_raw_bps"]].round(3).to_markdown(index=False), "",
             "2023 annual averages versus 2026 January-September averages; equal-weight means of country monthly means; eight fixed currencies. Same-date raw legs, 15-quote monthly minimum, no forward fill. The seven missing CAD 10Y historical quotes are retained from the old identical-ticker download with provenance.", "",
             "## Country-Level Accounting", "", countries[["currency", "tenor", "foreign_swap_bps", "foreign_govt_yield_bps", "foreign_swap_spread_bps", "x_govt_raw_bps"]].round(3).to_markdown(index=False), "",
             "A falling foreign spread means foreign government yields rise relative to the local swap benchmark, even when both outright yields fall. That increases the relative UST convenience measure, holding the other terms fixed. It need not imply an improvement in U.S. Treasury absolute safety or liquidity.", "",
             "## Endpoint and Baseline Sensitivity", "",
             sensitivity[sensitivity.excluded_currency.eq("none")][["baseline", "endpoint", "tenor", "x_govt_raw_bps", "foreign_swap_spread_bps", "negative_foreign_spread_changes"]].round(3).to_markdown(index=False), "",
             "Leave-one-currency-out results are retained in treasury_final_rebound_sensitivity.csv. These are descriptive accounting sensitivity checks, not independently identified causal effects or confidence intervals.", "",
             "## Supply and Inventory Associations", "", main_models[["tenor", "variable", "coef_bps_per_sd", "ci95_low", "ci95_high", "n_months", "last_month"]].round(3).to_markdown(index=False), "",
             "Annual-change regressions use calendar-aware HAC with 12 lags, exclude full windows crossing dealer/benchmark regimes, and control for MOVE, VIX, the dollar index, the U.S. yield and curve slope. The coefficients are conditional associations, not supply shocks. Net dealer positions measure inventory, not spare capacity. Supply-share and COVID-window exclusions are shown in the appendix; p-values are exploratory and not multiplicity-adjusted.", "",
             "## Interpretation", "",
             f"[BIS (2024)]({SOURCES['bis']}) discusses competing cash-bond absorption, financing/balance-sheet and swap receiving-demand channels for lower swap spreads. Our raw-leg split distinguishes accounting contributions, not these channels.", "",
             f"[ECB (2024)]({SOURCES['ecb']}) studies QT expectations and OIS term premia. QT therefore cannot be assumed to affect only government yields. A new causal claim would require a separate identification design.", "",
             "## Data Decision", "",
             "No additional Bloomberg download is required for this final descriptive presentation. Extending the supply regressions through September would require refreshed MOVE/VIX/dollar/yield/slope controls and updated official supply/inventory, but is not necessary for the stated endpoint. A causal foreign-supply/QT extension is future work, not completed by refreshing a spreadsheet.", "",
             "3M, EM NDS/CCS, JPY 20Y/30Y and NOK 1Y remain outside the validated final core. Raw Bloomberg inputs are licensed and should not be publicly redistributed.", "",
             "## Sources", "", *[f"- {key}: {url}" for key, url in SOURCES.items()]]
    (REPORT / "Treasury_Convenience_Final_Findings.md").write_text("\n".join(lines) + "\n")
    print(primary[["tenor", "x_govt_raw_bps", "x_rf_raw_bps", "us_swap_spread_bps", "foreign_swap_spread_bps"]].round(3).to_string(index=False))
    print("Complete months:", cutoff.date(), "; country-tenor months:", len(monthly))
    print(sensitivity[sensitivity.excluded_currency.eq("none")][["baseline", "endpoint", "tenor", "x_govt_raw_bps"]].round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
