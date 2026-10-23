"""Benchmark-definition sensitivity and a VIX-only weekly research first pass."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from src.analyze_treasury_interim import CLEAN, COHORT, FIGURE, ROOT, TABLE, plt
from src.refresh_treasury_5y10y import PANEL

OUTCOMES = ["x_govt_raw_bps", "x_rf_raw_bps", "us_swap_spread_bps", "foreign_swap_spread_bps"]
GROUPS = {"fixed8": COHORT, "exclude_NZD": [c for c in COHORT if c != "NZD"],
          "exclude_NZD_DKK": [c for c in COHORT if c not in {"NZD", "DKK"}],
          "overnight_metadata4": ["CAD", "CHF", "EUR", "GBP"], "term_metadata3": ["AUD", "DKK", "SEK"]}


def weekly_hac(model, dates: pd.DatetimeIndex, lags: int = 26) -> np.ndarray:
    weeks = dates.to_period("W-FRI").asi8
    assert len(np.unique(weeks)) == len(weeks)
    scores = np.zeros((int(weeks.max() - weeks.min() + 1), model.model.exog.shape[1]))
    scores[weeks - weeks.min()] = model.model.exog * np.asarray(model.resid)[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, len(scores) - 1) + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    n, k = model.model.exog.shape
    bread = np.asarray(model.normalized_cov_params)
    return bread @ meat @ bread * n / (n - k)


def weekly_panel(raw: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    selected = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT)].copy()
    assert not selected.duplicated(["date", "currency", "tenor"]).any()
    vix = vix.set_index("date").value.rename("vix")
    assert vix.index.is_unique and vix.gt(0).all()
    cutoff = min(selected.date.max(), vix.index.max(), pd.Timestamp("2026-09-30"))
    last_friday = cutoff - pd.Timedelta(days=(cutoff.weekday() - 4) % 7)
    results = []
    for tenor in ["5y", "10y"]:
        sub = selected[selected.tenor.eq(tenor)]
        wide = sub.pivot(index="date", columns="currency", values=OUTCOMES)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([OUTCOMES, COHORT]))
        common = wide.notna().all(axis=1) & wide.index.isin(vix.index)
        wide = wide.loc[common & (wide.index <= last_friday)]
        regimes = sub.pivot(index="date", columns="currency", values="regime").reindex(index=wide.index, columns=COHORT)
        daily = pd.DataFrame({metric: wide[metric].mean(axis=1) for metric in OUTCOMES})
        daily["vix"] = vix.reindex(daily.index)
        daily["signature"] = regimes.agg("|".join, axis=1)
        grouped = daily.resample("W-FRI")
        weekly = grouped[OUTCOMES + ["vix"]].mean()
        weekly["n_joint_days"] = grouped.size()
        weekly["n_regimes"] = grouped.signature.nunique()
        weekly["signature"] = grouped.signature.first()
        good = weekly.n_joint_days.ge(3) & weekly.n_regimes.eq(1)
        weekly.loc[~good, OUTCOMES + ["vix"]] = np.nan
        # Keep the calendar grid: a missing week must not become a two-week difference.
        weekly["dlog_vix"] = np.log(weekly.vix).diff()
        for metric in OUTCOMES:
            weekly[f"d_{metric}"] = weekly[metric].diff()
        weekly["valid_change"] = good & good.shift(1, fill_value=False) & weekly.signature.eq(weekly.signature.shift())
        weekly["tenor"] = tenor
        results.append(weekly.rename_axis("week").reset_index())
    return pd.concat(results, ignore_index=True)


def risk_models(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tenor, data in weekly.groupby("tenor"):
        data = data[data.valid_change & data.week.ge("2010-01-01")].set_index("week")
        windows = {"2010-2015": data.index.year <= 2015,
                   "2016-2020": (data.index.year >= 2016) & (data.index.year <= 2020),
                   "2021-latest": data.index.year >= 2021,
                   "post2023_interaction": np.ones(len(data), dtype=bool),
                   "recent_regime_interaction": data.signature.eq(data.signature.iloc[-1]).to_numpy()}
        for label, mask in windows.items():
            sample = data.loc[mask].dropna(subset=["dlog_vix", *[f"d_{m}" for m in OUTCOMES]])
            interaction = label.endswith("interaction")
            x = pd.DataFrame({"dlog_vix": sample.dlog_vix}, index=sample.index)
            if interaction:
                x["post2023"] = (sample.index >= "2023-01-01").astype(float)
                x["dlog_vix_post2023"] = x.dlog_vix * x.post2023
                if label == "post2023_interaction":
                    x = pd.concat([x, pd.get_dummies(sample.signature, prefix="regime", drop_first=True, dtype=float)], axis=1)
                assert x.post2023.eq(0).sum() >= 26 and x.post2023.eq(1).sum() >= 26
            x = sm.add_constant(x).astype(float)
            assert len(sample) >= 50 and np.linalg.matrix_rank(x) == x.shape[1]
            for metric in OUTCOMES:
                fitted = sm.OLS(sample[f"d_{metric}"], x).fit()
                covariance = weekly_hac(fitted, sample.index)
                contrasts = {"beta_vix": {"dlog_vix": 1.}}
                if interaction:
                    contrasts = {"pre2023_beta": {"dlog_vix": 1.},
                                 "post2023_beta": {"dlog_vix": 1., "dlog_vix_post2023": 1.},
                                 "beta_change": {"dlog_vix_post2023": 1.}}
                for name, weights in contrasts.items():
                    vector = np.array([weights.get(col, 0.) for col in x.columns])
                    beta = float(vector @ fitted.params)
                    se = float(np.sqrt(max(0., vector @ covariance @ vector)))
                    rows.append({"tenor": tenor, "model": label, "outcome": metric, "contrast": name,
                                 "beta_bps_per_log_vix": beta, "se_hac26": se,
                                 "ci95_low": beta - 1.96 * se, "ci95_high": beta + 1.96 * se,
                                 "p_exploratory": 2 * norm.sf(abs(beta / se)) if se else np.nan,
                                 "n_weeks": len(sample), "first_week": sample.index.min(),
                                 "last_week": sample.index.max(), "r_squared": fitted.rsquared,
                                 "n_pre2023": int((sample.index < "2023-01-01").sum()),
                                 "n_post2023": int((sample.index >= "2023-01-01").sum())})
    result = pd.DataFrame(rows)
    bridge = result.pivot(index=["tenor", "model", "contrast"], columns="outcome", values="beta_bps_per_log_vix")
    assert (bridge.x_govt_raw_bps - bridge.x_rf_raw_bps - bridge.us_swap_spread_bps + bridge.foreign_swap_spread_bps).abs().max() < 1e-8
    return result


def benchmark_sensitivity(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(CLEAN / "treasury_final_country_monthly.csv", parse_dates=["month"])
    changes = []
    for label, currencies in GROUPS.items():
        data = monthly[monthly.currency.isin(currencies)]
        assert data.groupby(["month", "tenor"]).size().eq(len(currencies)).all()
        aggregate = data.groupby(["month", "tenor"])[OUTCOMES].mean().reset_index()
        before = aggregate[aggregate.month.dt.year.eq(2023)].groupby("tenor")[OUTCOMES].mean()
        after = aggregate[aggregate.month.dt.year.eq(2026)].groupby("tenor")[OUTCOMES].mean()
        delta = (after - before).reset_index()
        delta["cohort"], delta["n_currencies"] = label, len(currencies)
        changes.append(delta)
    recent = raw[raw.is_selected_regime_by_public_break & raw.currency.eq("NZD") & raw.date.between("2023-01-01", "2026-09-30")].copy()
    recent["month"] = recent.date.dt.to_period("M").dt.to_timestamp("M")
    recent["adjustment_bps"] = recent.tenor_basis * recent.tenor_basis_sign
    nz = recent.groupby(["tenor", "month"])[OUTCOMES + ["adjustment_bps"]].mean().reset_index()
    nz_delta = nz[nz.month.dt.year.eq(2026)].groupby("tenor")[OUTCOMES + ["adjustment_bps"]].mean() - nz[nz.month.dt.year.eq(2023)].groupby("tenor")[OUTCOMES + ["adjustment_bps"]].mean()
    nz_delta["aligned_x_rf_change_bps"] = nz_delta.x_rf_raw_bps + nz_delta.adjustment_bps
    nz_delta["aligned_foreign_spread_change_bps"] = nz_delta.foreign_swap_spread_bps + nz_delta.adjustment_bps
    nz_delta["fixed8_component_reallocation_bps"] = nz_delta.adjustment_bps / len(COHORT)
    # Relabeling the benchmark reallocates two Eq.6 components, not Treasury convenience.
    np.testing.assert_allclose(nz_delta.x_govt_raw_bps, nz_delta.aligned_x_rf_change_bps + nz_delta.us_swap_spread_bps - nz_delta.aligned_foreign_spread_change_bps, atol=1e-8)
    inputs_path = TABLE / "foreign_swap_spread_refresh_inputs.csv"
    if not inputs_path.exists():
        inputs_path = ROOT / "metadata" / inputs_path.name
    inputs = pd.read_csv(inputs_path).fillna("")
    audit = inputs[inputs.type.eq("IRS") & inputs.currency.ne("USD")].copy()
    audit["interpretation"] = audit.currency.map({c: "Overnight benchmark per supplied metadata" for c in GROUPS["overnight_metadata4"]} | {c: "Term bank benchmark per supplied metadata" for c in GROUPS["term_metadata3"]} | {"NZD": "REVIEW: instrument name says OCR; floating_leg says Bank Bill; tenor adjustment also used"})
    return pd.concat(changes, ignore_index=True), nz_delta.reset_index(), audit


def make_figures(sensitivity: pd.DataFrame, models: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = {"fixed8": "Fixed eight", "exclude_NZD": "Without NZD", "exclude_NZD_DKK": "Without NZD/DKK",
              "overnight_metadata4": "Overnight four", "term_metadata3": "Term-bank three"}
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = sensitivity[sensitivity.tenor.eq(tenor)].set_index("cohort").loc[list(GROUPS)]
        ax.barh([labels[x] for x in sub.index], sub.x_govt_raw_bps, color="#246b91")
        ax.axvline(0, color="#444444", lw=.8)
        ax.set_title(tenor.upper())
        ax.set_xlabel("Convenience change (bps)")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("2023 average to 2026 Jan-Sep: rebound under benchmark-cohort exclusions")
    fig.tight_layout()
    fig.savefig(FIGURE / "post_meeting_benchmark_sensitivity.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = models[models.tenor.eq(tenor) & models.outcome.eq("x_govt_raw_bps") & models.contrast.eq("beta_vix")]
        ax.errorbar(sub.model, sub.beta_bps_per_log_vix, yerr=[sub.beta_bps_per_log_vix - sub.ci95_low, sub.ci95_high - sub.beta_bps_per_log_vix], fmt="o", color="#246b91", capsize=5)
        ax.axhline(0, color="#444444", lw=.8)
        ax.set_title(tenor.upper())
        ax.set_xlabel("Weekly sample period")
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Change in Treasury convenience per log VIX change (bps)")
    fig.suptitle("VIX-only risk sensitivity; 95% calendar-HAC intervals, 26 weeks")
    fig.tight_layout()
    fig.savefig(FIGURE / "post_meeting_vix_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> dict:
    raw = pd.read_csv(PANEL, parse_dates=["date"], low_memory=False)
    coverage = pd.read_csv(TABLE / "downloaded_bloomberg_ticker_coverage.csv")
    control_coverage = coverage[coverage.workbook.eq("macro_controls")].copy()
    control_coverage[["ticker", "variable", "first_valid_date", "last_valid_date", "n_numeric"]].to_csv(TABLE / "post_meeting_existing_risk_coverage.csv", index=False)
    parts = []
    for chunk in pd.read_csv(CLEAN / "downloaded_bloomberg_values_long.csv", usecols=["workbook", "date", "variable", "value"], chunksize=250_000):
        parts.append(chunk[chunk.workbook.eq("macro_controls") & chunk.variable.eq("vix_index")][["date", "value"]])
    vix = pd.concat(parts, ignore_index=True)
    vix["date"] = pd.to_datetime(vix.date)
    assert not vix.date.duplicated().any()
    sensitivity, nz_delta, audit = benchmark_sensitivity(raw)
    sensitivity.to_csv(TABLE / "post_meeting_benchmark_sensitivity.csv", index=False)
    nz_delta.to_csv(TABLE / "post_meeting_nzd_component_sensitivity.csv", index=False)
    audit.to_csv(TABLE / "post_meeting_benchmark_definition_audit.csv", index=False)
    weekly = weekly_panel(raw, vix)
    weekly.to_csv(CLEAN / "post_meeting_weekly_risk_panel.csv", index=False)
    models = risk_models(weekly)
    models.to_csv(TABLE / "post_meeting_vix_risk_models.csv", index=False)
    make_figures(sensitivity, models)
    summary = {"updated": "2026-10-23", "scope": "Existing data only; VIX-only diagnostic, not complete Table 5 or the HY extension",
               "risk_last_week": str(weekly.week.max().date()), "vix_last_observation": str(vix.date.max().date()),
               "nzd_definition_status": "REVIEW: OCR-name versus Bank-Bill metadata conflict; no raw formula changed",
               "all_cohort_rebounds_positive": bool(sensitivity.x_govt_raw_bps.gt(0).all()),
               "all_cohort_foreign_spreads_decline": bool(sensitivity.foreign_swap_spread_bps.lt(0).all()),
               "risk_model_fits": len(models[["tenor", "model", "outcome"]].drop_duplicates()),
               "risk_coefficient_rows": len(models), "missing_series": ["SPX Index", "LUACOAS Index", "LF98OAS Index"],
               "caveats": ["Not exact paper replication: fixed-eight means, joint-date weekly means, VIX only.",
                           "A positive level rebound does not establish improved risk sensitivity or absolute safe-haven quality.",
                           "Pre/post interactions are conditional associations; regime and episode composition can confound them.",
                           "HAC 26-week intervals and p-values are exploratory and not multiplicity adjusted."]}
    (TABLE / "post_meeting_research_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
