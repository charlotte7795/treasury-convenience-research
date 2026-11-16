"""Country accounting for the existing fixed-rule, matched-calendar stress test."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analyze_post_meeting_research import weekly_hac
from src.analyze_risk_properties import OUTCOMES
from src.analyze_treasury_interim import CLEAN, COHORT, FIGURE, ROOT, TABLE, plt
from src.refresh_treasury_5y10y import PANEL

METRICS = [*OUTCOMES, "foreign_swap_bps", "foreign_govt_yield_bps"]
WEIGHT = 1 / len(COHORT)


def make_country_weekly(raw, market, baseline, flags):
    selected = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT)].copy()
    assert not selected.duplicated(["date", "currency", "tenor"]).any()
    selected["foreign_swap_bps"] = 100 * selected.foreign_swap
    selected["foreign_govt_yield_bps"] = 100 * selected.foreign_govt_yield
    parts, reconciliation = [], []
    for tenor in ["5y", "10y"]:
        reference = baseline[baseline.cohort.eq("fixed8") & baseline.tenor.eq(tenor)].set_index("week")
        sub = selected[selected.tenor.eq(tenor)]
        wide = sub.pivot(index="date", columns="currency", values=METRICS)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([METRICS, COHORT]))
        dates = wide.index[wide[OUTCOMES].notna().all(axis=1)].intersection(market.dropna().index)
        dates = dates[(dates >= "2010-01-01") & (dates <= reference.index.max())]
        assert wide.loc[dates].notna().all().all()
        regimes = sub.pivot(index="date", columns="currency", values="regime").loc[dates, COHORT].agg("|".join, axis=1)
        counts = regimes.resample("W-FRI").size()
        pd.testing.assert_index_equal(counts.index.rename("week"), reference.index)
        np.testing.assert_array_equal(counts, reference.n_joint_days)
        good = counts.ge(3) & regimes.resample("W-FRI").nunique().eq(1)
        levels = wide.loc[dates].resample("W-FRI").mean()
        levels.loc[~good, :] = np.nan
        changes = levels.diff()
        changes.loc[~reference.valid_change.to_numpy(), :] = np.nan
        for metric in OUTCOMES:
            error = changes[metric].mean(axis=1) - reference[f"d_{metric}"].where(reference.valid_change)
            assert error.abs().max() < 1e-8
            reconciliation.append({"tenor": tenor, "metric": metric, "max_weekly_mean_error_bps": error.abs().max()})
        for currency in COHORT:
            country = levels.xs(currency, level=1, axis=1).copy()
            for metric in METRICS:
                country[f"d_{metric}"] = changes[metric, currency]
            country["valid_change"] = reference.valid_change
            country["n_joint_days"] = counts
            country["currency"], country["tenor"] = currency, tenor
            country = country.join(flags, how="left", validate="one_to_one").loc["2023-01-01":]
            parts.append(country.rename_axis("week").reset_index())
    panel = pd.concat(parts, ignore_index=True)
    complete = panel[panel.valid_change].dropna(subset=["stress", *[f"d_{m}" for m in METRICS]])
    assert complete.groupby(["tenor", "week"]).currency.nunique().eq(len(COHORT)).all()
    assert (complete.d_x_govt_raw_bps - complete.d_x_rf_raw_bps - complete.d_us_swap_spread_bps + complete.d_foreign_swap_spread_bps).abs().max() < 1e-8
    assert (complete.d_foreign_swap_spread_bps - complete.d_foreign_swap_bps + complete.d_foreign_govt_yield_bps).abs().max() < 1e-8
    # The recent U.S. leg is shared; it is not eight independent country shocks.
    assert complete.groupby(["tenor", "week"]).d_us_swap_spread_bps.agg(lambda s: s.max() - s.min()).abs().max() < 1e-8
    return panel, pd.DataFrame(reconciliation)


def estimate_stress(sample, metric):
    y = sample[f"d_{metric}"]
    model = sm.OLS(y, sm.add_constant(sample[["stress"]])).fit()
    covariance = weekly_hac(model, sample.index)
    high, ordinary = y[sample.stress.eq(1)], y[sample.stress.eq(0)]
    leave_one = (high.sum() - high) / (len(high) - 1) - ordinary.mean()
    rows = []
    for contrast, vector in [("stress_mean", [1., 1.]), ("ordinary_mean", [1., 0.]), ("stress_minus_ordinary", [0., 1.])]:
        vector = np.array(vector)
        estimate = float(vector @ model.params)
        se = float(np.sqrt(vector @ covariance @ vector))
        rows.append({"metric": metric, "contrast": contrast, "estimate_bps": estimate,
                     "hac26_ci_low": estimate - 1.96 * se, "hac26_ci_high": estimate + 1.96 * se,
                     "n_stress": len(high), "n_ordinary": len(ordinary),
                     "leave_one_stress_out_min": leave_one.min() if contrast == "stress_minus_ordinary" else np.nan,
                     "leave_one_stress_out_max": leave_one.max() if contrast == "stress_minus_ordinary" else np.nan})
    return rows


def country_estimates(panel):
    rows, events = [], []
    for (tenor, currency), country in panel.groupby(["tenor", "currency"]):
        sample = country[country.valid_change].set_index("week").dropna(subset=["stress", *[f"d_{m}" for m in METRICS]])
        for metric in METRICS:
            rows.extend({"tenor": tenor, "currency": currency, **row} for row in estimate_stress(sample, metric))
        stress = sample[sample.stress.eq(1)].copy()
        stress["ordinary_mean_bps"] = sample.loc[sample.stress.eq(0), "d_x_govt_raw_bps"].mean()
        stress["centered_change_bps"] = stress.d_x_govt_raw_bps - stress.ordinary_mean_bps
        stress["aggregate_contribution_bps"] = stress.centered_change_bps * WEIGHT / len(stress)
        events.append(stress.reset_index())
    estimates = pd.DataFrame(rows)
    estimates["equal_weight_contribution_bps"] = estimates.estimate_bps * WEIGHT
    return estimates, pd.concat(events, ignore_index=True)


def accounting(estimates, previous):
    rows, ledger = [], []
    for (tenor, contrast), group in estimates.groupby(["tenor", "contrast"]):
        wide = group.pivot(index="currency", columns="metric", values="estimate_bps").loc[COHORT]
        common_us = wide.us_swap_spread_bps.iloc[0]
        assert np.ptp(wide.us_swap_spread_bps) < 1e-8
        target = previous[previous.cohort.eq("fixed8") & previous.tenor.eq(tenor) & previous.period.eq("post2023")
                          & previous.outcome.eq("x_govt_raw_bps") & previous.contrast.eq(contrast)].estimate_bps.item()
        reconstructed = common_us + (wide.x_rf_raw_bps - wide.foreign_swap_spread_bps).mean()
        assert abs(reconstructed - target) < 1e-8
        assert abs(wide.x_govt_raw_bps.mean() - target) < 1e-8
        ledger.append({"tenor": tenor, "contrast": contrast, "previous_aggregate_bps": target,
                       "sum_weighted_pair_bps": wide.x_govt_raw_bps.mean(), "common_us_bps": common_us,
                       "dollar_bps": wide.x_rf_raw_bps.mean(), "minus_foreign_spread_bps": -wide.foreign_swap_spread_bps.mean(),
                       "minus_foreign_swap_bps": -wide.foreign_swap_bps.mean(),
                       "plus_foreign_govt_bps": wide.foreign_govt_yield_bps.mean(),
                       "reconstructed_bps": reconstructed, "residual_bps": target - reconstructed})
        if contrast != "stress_minus_ordinary":
            continue
        for currency, data in wide.iterrows():
            rows.append({"tenor": tenor, "currency": currency, "pair_contrast_bps": data.x_govt_raw_bps,
                         "weighted_pair_contribution_bps": data.x_govt_raw_bps * WEIGHT,
                         "common_us_allocation_bps": common_us * WEIGHT,
                         "dollar_contribution_bps": data.x_rf_raw_bps * WEIGHT,
                         "minus_foreign_spread_contribution_bps": -data.foreign_swap_spread_bps * WEIGHT,
                         "country_specific_contribution_bps": (data.x_rf_raw_bps - data.foreign_swap_spread_bps) * WEIGHT,
                         "minus_foreign_swap_contribution_bps": -data.foreign_swap_bps * WEIGHT,
                         "plus_foreign_govt_contribution_bps": data.foreign_govt_yield_bps * WEIGHT})
    return pd.DataFrame(rows), pd.DataFrame(ledger)


def leave_one_country(panel):
    rows = []
    valid = panel[panel.valid_change].dropna(subset=["stress", "d_x_govt_raw_bps"])
    for tenor, group in valid.groupby("tenor"):
        wide = group.pivot(index="week", columns="currency", values="d_x_govt_raw_bps").loc[:, COHORT]
        stress = group.groupby("week").stress.first()
        assert wide.notna().all().all()
        for omitted in COHORT:
            sample = pd.DataFrame({"d_x_govt_raw_bps": wide.drop(columns=omitted).mean(axis=1), "stress": stress})
            row = estimate_stress(sample, "x_govt_raw_bps")[-1]
            rows.append({"tenor": tenor, "omitted_currency": omitted, **row})
    return pd.DataFrame(rows)


def public_overlap_check(raw, market, baseline, flags, public):
    rows = []
    selected = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT)]
    for tenor in ["5y", "10y"]:
        sub = selected[selected.tenor.eq(tenor)]
        wide = sub.pivot(index="date", columns="currency", values=OUTCOMES)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([OUTCOMES, COHORT]))
        dates = wide.dropna().index.intersection(market.dropna().index)
        dates = dates[(dates >= "2022-12-01") & (dates <= baseline.week.max())]
        p = public[public.tenor.eq(tenor) & public.currency.isin(COHORT)].pivot(index="date", columns="currency", values="cip_govt")
        p = p.reindex(index=dates, columns=COHORT).dropna()
        r = wide.loc[p.index, "x_govt_raw_bps"]
        good = p.resample("W-FRI").size().ge(3)
        p, r = p.resample("W-FRI").mean(), r.resample("W-FRI").mean()
        p.loc[~good], r.loc[~good] = np.nan, np.nan
        dp, dr = p.diff(), r.diff()
        reference = baseline[baseline.cohort.eq("fixed8") & baseline.tenor.eq(tenor)].set_index("week")
        valid = reference.valid_change.reindex(dp.index, fill_value=False) & (dp.index >= "2023-01-01")
        valid &= dp.notna().all(axis=1) & dr.notna().all(axis=1)
        stress = flags.stress.reindex(dp.index)
        valid &= stress.notna()
        for currency in COHORT:
            raw_contrast = dr.loc[valid & stress.eq(1), currency].mean() - dr.loc[valid & stress.eq(0), currency].mean()
            public_contrast = dp.loc[valid & stress.eq(1), currency].mean() - dp.loc[valid & stress.eq(0), currency].mean()
            rows.append({"tenor": tenor, "currency": currency, "n_stress": int((valid & stress.eq(1)).sum()),
                         "n_ordinary": int((valid & stress.eq(0)).sum()), "first_week": dp.index[valid].min(),
                         "last_week": dp.index[valid].max(), "raw_contrast_bps": raw_contrast,
                         "public_contrast_bps": public_contrast, "contrast_gap_bps": raw_contrast - public_contrast,
                         "weekly_change_mae_bps": (dr.loc[valid, currency] - dp.loc[valid, currency]).abs().mean()})
    return pd.DataFrame(rows)


def figures(estimates, attribution, events, ledger):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
    for tenor, ax in zip(["5y", "10y"], axes):
        data = estimates[estimates.tenor.eq(tenor) & estimates.metric.eq("x_govt_raw_bps")
                         & estimates.contrast.eq("stress_minus_ordinary")].sort_values("estimate_bps")
        ax.barh(data.currency, data.estimate_bps, color=np.where(data.estimate_bps < 0, "#246b91", "#426b50"), alpha=.75)
        ax.errorbar(data.estimate_bps, np.arange(len(data)), xerr=[data.estimate_bps - data.hac26_ci_low, data.hac26_ci_high - data.estimate_bps],
                    fmt="none", ecolor="#333333", capsize=3)
        ax.axvline(0, color="#555555", lw=.8)
        ax.invert_yaxis()
        ax.set_title(tenor.upper())
        ax.set_xlabel("Stress minus ordinary weekly change (bps)")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("2023 to September 2026: relative Treasury convenience by pair\nSame 13 stress weeks per tenor; exploratory 95% HAC-26 intervals")
    fig.tight_layout()
    fig.savefig(FIGURE / "country_stress_pair_contrasts.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, sharey=True)
    for tenor, ax in zip(["5y", "10y"], axes):
        data = attribution[attribution.tenor.eq(tenor)].set_index("currency").loc[COHORT]
        a, b = data.dollar_contribution_bps, data.minus_foreign_spread_contribution_bps
        y = np.arange(len(COHORT))
        ax.barh(y, a, color="#426b50", label="Dollar-convenience proxy")
        ax.barh(y, b, left=np.where(a * b > 0, a, 0.), color="#a65b3e", label="Minus foreign swap spread")
        ax.plot(a + b, y, "kD", markersize=4, label="Country-specific net")
        common_us = ledger[ledger.tenor.eq(tenor) & ledger.contrast.eq("stress_minus_ordinary")].common_us_bps.item()
        ax.barh([8.5], [common_us], color="#777777", label="Common U.S. swap spread")
        ax.set_yticks([*y, 8.5], [*COHORT, "U.S. common leg"])
        ax.axvline(0, color="#555555", lw=.8)
        ax.set_title(tenor.upper())
        ax.set_xlabel("Contribution to fixed-eight contrast (bps)")
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=2)
    fig.suptitle("Country-specific terms plus one shared U.S. term\nCountry terms weighted 1/8; signed point accounting, not causal attribution")
    fig.tight_layout(rect=[0, .1, 1, 1])
    fig.savefig(FIGURE / "country_stress_component_contributions.png", dpi=180)
    plt.close(fig)

    limit = events.centered_change_bps.abs().max()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for tenor, ax in zip(["5y", "10y"], axes):
        data = events[events.tenor.eq(tenor)].pivot(index="week", columns="currency", values="centered_change_bps").loc[:, COHORT]
        im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(np.arange(len(COHORT)), COHORT)
        ax.set_yticks(np.arange(len(data)), data.index.strftime("%Y-%m-%d"))
        ax.set_title(tenor.upper())
        for (i, j), value in np.ndenumerate(data.to_numpy()):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > .6 * limit else "black")
    fig.colorbar(im, ax=axes, label="Weekly convenience change minus country ordinary-week mean (bps)", shrink=.8)
    fig.suptitle("All observed recent stress weeks, without event selection\nUnweighted pair changes; not country contributions or Treasury returns")
    fig.savefig(FIGURE / "country_stress_week_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    raw = pd.read_csv(PANEL, parse_dates=["date"], low_memory=False)
    values = pd.read_csv(CLEAN / "credit_risk_downloaded_values.csv", parse_dates=["date"])
    market = values[values.file.eq("Risk_Extension_Bloomberg_BDH_Template.xlsx")].pivot(index="date", columns="variable", values="value")
    baseline = pd.read_csv(CLEAN / "credit_risk_weekly_panel.csv", parse_dates=["week"])
    flags = pd.read_csv(CLEAN / "risk_properties_stress_calendar.csv", parse_dates=["date"]).set_index("date")
    previous = pd.read_csv(TABLE / "risk_properties_stress_results.csv")
    public = pd.read_csv(CLEAN / "public_cip_cleaned.csv", parse_dates=["date"])
    panel, reconciliation = make_country_weekly(raw, market, baseline, flags)
    panel.to_csv(CLEAN / "country_stress_weekly_panel.csv", index=False)
    reconciliation.to_csv(TABLE / "country_stress_weekly_reconciliation.csv", index=False)
    estimates, events = country_estimates(panel)
    attribution, ledger = accounting(estimates, previous)
    omitted = leave_one_country(panel)
    for tenor in ["5y", "10y"]:
        event_sum = events[events.tenor.eq(tenor)].aggregate_contribution_bps.sum()
        target = ledger[ledger.tenor.eq(tenor) & ledger.contrast.eq("stress_minus_ordinary")].previous_aggregate_bps.item()
        assert abs(event_sum - target) < 1e-8
        ex_nzd = previous[previous.tenor.eq(tenor) & previous.cohort.eq("exclude_NZD") & previous.period.eq("post2023")
                          & previous.outcome.eq("x_govt_raw_bps") & previous.contrast.eq("stress_minus_ordinary")].estimate_bps.item()
        assert abs(omitted[omitted.tenor.eq(tenor) & omitted.omitted_currency.eq("NZD")].estimate_bps.item() - ex_nzd) < 1e-8
    estimates.to_csv(TABLE / "country_stress_estimates.csv", index=False)
    attribution.to_csv(TABLE / "country_stress_attribution.csv", index=False)
    ledger.to_csv(TABLE / "country_stress_accounting_ledger.csv", index=False)
    omitted.to_csv(TABLE / "country_stress_leave_one_country_out.csv", index=False)
    events.to_csv(TABLE / "country_stress_event_contributions.csv", index=False)
    overlap = public_overlap_check(raw, market, baseline, flags, public)
    overlap.to_csv(TABLE / "country_stress_public_overlap_check.csv", index=False)
    figures(estimates, attribution, events, ledger)
    summary = {"updated": "2026-11-02", "currencies": COHORT, "weight": WEIGHT,
               "start_week": str(panel.week.min().date()), "end_week": str(panel.week.max().date()),
               "country_week_rows": len(panel), "country_estimate_rows": len(estimates),
               "max_weekly_reconciliation_error_bps": reconciliation.max_weekly_mean_error_bps.max(),
               "max_accounting_residual_bps": ledger.residual_bps.abs().max(),
               "stress_rule": "Reuse risk_properties_stress_calendar.csv unchanged; no threshold refit",
               "inference": "HAC-26 country intervals are exploratory, unadjusted for multiple comparisons; ranking is descriptive",
               "caveats": ["Same fixed-eight joint dates; deleting a country never expands the calendar.",
                           "The common U.S. leg is counted once, not eight independent observations.",
                           "Country-specific includes pairwise dollar funding; it does not isolate a local causal shock.",
                           "Thirteen stress weeks and unresolved benchmark conventions limit mechanism claims."],
               "input_receipts": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for path in [PANEL, CLEAN / "credit_risk_downloaded_values.csv", CLEAN / "credit_risk_weekly_panel.csv",
                                               CLEAN / "risk_properties_stress_calendar.csv", TABLE / "risk_properties_stress_results.csv",
                                               CLEAN / "public_cip_cleaned.csv"]]}
    (TABLE / "country_stress_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
