"""Use the November Bloomberg exports for matched-sample credit-risk tests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl import load_workbook
from scipy.stats import norm

from src.analyze_post_meeting_research import OUTCOMES, weekly_hac
from src.analyze_treasury_interim import CLEAN, COHORT, FIGURE, ROOT, TABLE, plt
from src.refresh_treasury_5y10y import PANEL

SNAPSHOT = ROOT / "data/refresh/risk_extension_2026-11-02"
FILES = ["Risk_Extension_Bloomberg_BDH_Template.xlsx", "NZD_Benchmark_Check_Bloomberg_BDH_Template.xlsx"]
FACTORS = ["dlog_vix", "spx_return", "d_ig_bps", "d_hy_bps", "dlog_move", "dlog_dollar"]
SPECS = {"VIX_only": FACTORS[:1], "VIX_equity_IG": FACTORS[:3], "plus_HY": FACTORS[:4],
         "plus_HY_controls": FACTORS}


def read_snapshot(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    manifest = list(wb["01_Inputs"].iter_rows(min_row=2, max_row=7, values_only=True))
    metadata = {r[0]: r for r in wb["02_Verification"].iter_rows(min_row=2, max_row=7, values_only=True)}
    values, audit = [], []
    start, stop = [pd.Timestamp(wb["00_Control"][cell].value) for cell in ["B2", "B3"]]
    for sheet, ticker, variable, unit, *_ in manifest:
        rows = list(wb[sheet].iter_rows(values_only=True))
        assert rows[0][1] == ticker and rows[1][1] == variable and rows[2][1] == "PX_LAST"
        observations, errors = [], []
        for row in rows[4:]:
            dt, quote = row[:2]
            if dt is None and quote is None:
                continue
            if (isinstance(dt, (date, datetime)) and isinstance(quote, (int, float))
                    and not isinstance(quote, bool) and np.isfinite(quote)):
                observations.append((pd.Timestamp(dt), float(quote)))
            else:
                errors.append(str((dt, quote)))
        data = pd.DataFrame(observations, columns=["date", "value"])
        assert not data.empty and not errors, (sheet, errors[:3])
        assert data.date.is_unique and data.date.is_monotonic_increasing
        assert data.date.between(start, stop).all() and data.date.dt.weekday.lt(5).all()
        meta = metadata[ticker]
        audit.append({"file": path.name, "sheet": sheet, "ticker": ticker, "variable": variable,
                      "live_name": meta[1], "live_id": meta[2], "expected_unit": unit,
                      "manual_definition": meta[3], "manual_unit": meta[4], "manual_multiplier": meta[5],
                      "n_numeric": len(data), "first_date": data.date.min(), "last_date": data.date.max(),
                      "minimum": data.value.min(), "median": data.value.median(), "maximum": data.value.max(),
                      "n_errors": len(errors), "n_missing_weekdays": len(pd.bdate_range(start, stop).difference(data.date)),
                      "unit_status": "Manual DES/unit cells blank; provisional recorded convention" if meta[4] is None else "Manual unit supplied"})
        data["variable"], data["ticker"], data["file"] = variable, ticker, path.name
        values.append(data)
    wb.close()
    return pd.concat(values, ignore_index=True), pd.DataFrame(audit)


def make_weekly(raw, market):
    selected = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT)].copy()
    assert not selected.duplicated(["date", "currency", "tenor"]).any()
    assert market.index.is_unique and market.stack().gt(0).all()
    stop = min(selected.date.max(), market.index.max(), pd.Timestamp("2026-09-30"))
    stop -= pd.Timedelta(days=(stop.weekday() - 4) % 7)
    parts = []
    for tenor in ["5y", "10y"]:
        sub = selected[selected.tenor.eq(tenor)]
        wide = sub.pivot(index="date", columns="currency", values=OUTCOMES)
        wide = wide.reindex(columns=pd.MultiIndex.from_product([OUTCOMES, COHORT]))
        dates = wide.index[wide.notna().all(axis=1) & (wide.index <= stop)].intersection(market.dropna().index)
        dates = dates[dates >= "2010-01-01"]
        regimes = sub.pivot(index="date", columns="currency", values="regime").reindex(index=dates, columns=COHORT)
        for cohort, currencies in {"fixed8": COHORT, "exclude_NZD": [c for c in COHORT if c != "NZD"]}.items():
            daily = pd.DataFrame({metric: wide.loc[dates, (metric, currencies)].mean(axis=1) for metric in OUTCOMES})
            daily = daily.join(market, how="left", validate="one_to_one")
            # Both cohorts retain exactly the same daily dates and regime exclusions.
            daily["signature"] = regimes.agg("|".join, axis=1)
            grouped = daily.resample("W-FRI")
            numeric = [*OUTCOMES, *market.columns]
            weekly = grouped[numeric].mean()
            weekly["n_joint_days"] = grouped.size()
            weekly["signature"] = grouped.signature.first()
            good = weekly.n_joint_days.ge(3) & grouped.signature.nunique().eq(1)
            weekly.loc[~good, numeric] = np.nan
            for source, target in [("vix_index", "dlog_vix"), ("spx_index", "spx_return"),
                                   ("move_index", "dlog_move"), ("bbdxy_index", "dlog_dollar")]:
                weekly[target] = np.log(weekly[source]).diff()
            for source, target in [("ig_oas", "d_ig_bps"), ("hy_oas", "d_hy_bps")]:
                weekly[target] = 100 * weekly[source].diff()
            for metric in OUTCOMES:
                weekly[f"d_{metric}"] = weekly[metric].diff()
            weekly["valid_change"] = good & good.shift(1, fill_value=False) & weekly.signature.eq(weekly.signature.shift())
            weekly["cohort"], weekly["tenor"] = cohort, tenor
            parts.append(weekly.rename_axis("week").reset_index())
    return pd.concat(parts, ignore_index=True)


def fit_models(weekly):
    coefficients, fits = [], []
    for (cohort, tenor), frame in weekly.groupby(["cohort", "tenor"]):
        frame = frame[frame.valid_change].set_index("week").dropna(subset=FACTORS + [f"d_{m}" for m in OUTCOMES])
        windows = {"2010-2015": frame.index.year <= 2015,
                   "2016-2020": (frame.index.year >= 2016) & (frame.index.year <= 2020),
                   "2021-2025": (frame.index.year >= 2021) & (frame.index.year <= 2025),
                   "post2023": frame.index.year >= 2023,
                   "full_interaction": np.ones(len(frame), dtype=bool),
                   "recent_regime_interaction": frame.signature.eq(frame.signature.iloc[-1]).to_numpy()}
        for window, mask in windows.items():
            sample = frame.loc[mask]
            interaction = window.endswith("interaction")
            assert len(sample) >= 50
            for spec, factors in SPECS.items():
                if spec == "plus_HY_controls" and window != "post2023":
                    continue
                x = sample[factors].copy()
                if interaction:
                    x["post2023"] = (sample.index >= "2023-01-01").astype(float)
                    for factor in factors:
                        x[f"{factor}_post2023"] = x[factor] * x.post2023
                    if window == "full_interaction":
                        x = pd.concat([x, pd.get_dummies(sample.signature, prefix="regime", drop_first=True, dtype=float)], axis=1)
                    assert x.post2023.eq(0).sum() >= 26 and x.post2023.eq(1).sum() >= 50
                x = sm.add_constant(x).astype(float)
                assert np.linalg.matrix_rank(x) == x.shape[1]
                for outcome in OUTCOMES:
                    model = sm.OLS(sample[f"d_{outcome}"], x).fit()
                    record = {"cohort": cohort, "tenor": tenor, "window": window, "spec": spec, "outcome": outcome,
                              "n_weeks": len(sample), "first_week": sample.index.min(), "last_week": sample.index.max(),
                              "n_pre2023": int((sample.index < "2023-01-01").sum()),
                              "n_post2023": int((sample.index >= "2023-01-01").sum())}
                    fits.append({**record, "r_squared": model.rsquared, "adj_r_squared": model.rsquared_adj,
                                 "rss": model.ssr, "condition_standardized": np.linalg.cond(sm.add_constant((sample[factors] - sample[factors].mean()) / sample[factors].std()))})
                    for lags in ([13, 26] if window == "post2023" else [26]):
                        covariance = weekly_hac(model, sample.index, lags)
                        for factor in factors:
                            contrasts = {"beta": {factor: 1.}} if not interaction else {
                                "pre2023_beta": {factor: 1.}, "post2023_beta": {factor: 1., f"{factor}_post2023": 1.},
                                "beta_change": {f"{factor}_post2023": 1.}}
                            for contrast, weights in contrasts.items():
                                vector = np.array([weights.get(col, 0.) for col in x.columns])
                                beta = float(vector @ model.params)
                                se = float(np.sqrt(max(0., vector @ covariance @ vector)))
                                coefficients.append({**record, "factor": factor, "contrast": contrast, "hac_weeks": lags,
                                                     "beta": beta, "se": se, "ci95_low": beta - 1.96 * se,
                                                     "ci95_high": beta + 1.96 * se, "p_exploratory": 2 * norm.sf(abs(beta / se))})
    coefficients, fits = pd.DataFrame(coefficients), pd.DataFrame(fits)
    keys = ["cohort", "tenor", "window", "spec", "factor", "contrast", "hac_weeks"]
    bridge = coefficients.pivot(index=keys, columns="outcome", values="beta")
    assert (bridge.x_govt_raw_bps - bridge.x_rf_raw_bps - bridge.us_swap_spread_bps + bridge.foreign_swap_spread_bps).abs().max() < 1e-8
    base = fits[fits.spec.eq("VIX_equity_IG")].drop(columns="spec")
    full = fits[fits.spec.eq("plus_HY")].drop(columns="spec")
    incremental = base.merge(full, on=["cohort", "tenor", "window", "outcome"], suffixes=("_base", "_hy"), validate="one_to_one")
    assert incremental.n_weeks_base.eq(incremental.n_weeks_hy).all()
    assert incremental.first_week_base.eq(incremental.first_week_hy).all() and incremental.last_week_base.eq(incremental.last_week_hy).all()
    incremental["delta_r_squared"] = incremental.r_squared_hy - incremental.r_squared_base
    incremental["partial_r_squared_hy"] = 1 - incremental.rss_hy / incremental.rss_base
    assert incremental.delta_r_squared.ge(-1e-10).all()
    return coefficients, fits, incremental


def nzd_checks(values, baseline):
    wide = values.pivot(index="date", columns="variable", values="value")
    rows, daily = [], []
    for tenor, suffix in [("5y", "5"), ("10y", "10")]:
        names = [f"nzd_sofr_irs_{tenor}_ndso{suffix}", f"nzd_bank_swap_{tenor}_check", f"nzd_sofr_tbs_{tenor}_nzbbco{suffix}"]
        sub = wide[names].dropna().copy()
        sub.columns = ["ocr_percent", "bank_percent", "basis_bps"]
        sub["bank_minus_ocr_bps"] = 100 * (sub.bank_percent - sub.ocr_percent)
        sub["gap_minus_basis_bps"] = sub.bank_minus_ocr_bps - sub.basis_bps
        sub["gap_plus_basis_bps"] = sub.bank_minus_ocr_bps + sub.basis_bps
        sub["tenor"] = tenor
        daily.append(sub.reset_index())
        rows.append({"tenor": tenor, "n_joint_dates": len(sub), "first_date": sub.index.min(), "last_date": sub.index.max(),
                     "gap_basis_corr": sub.bank_minus_ocr_bps.corr(sub.basis_bps),
                     "mae_same_sign_bps": sub.gap_minus_basis_bps.abs().mean(),
                     "mean_same_sign_gap_bps": sub.gap_minus_basis_bps.mean(),
                     "mae_opposite_sign_bps": sub.gap_plus_basis_bps.abs().mean(),
                     "definition_status": "BDP NAME confirms OIS / 3M distinction; detailed DES conventions still unconfirmed"})
    # Reconcile the re-downloaded OIS/basis legs without overwriting the validated panel.
    revisions = []
    for tenor, suffix in [("5y", "5"), ("10y", "10")]:
        old = baseline[baseline.currency.eq("NZD") & baseline.tenor.eq(tenor) & baseline.is_selected_regime_by_public_break].set_index("date")
        for variable, source, scale in [(f"nzd_sofr_irs_{tenor}_ndso{suffix}", "foreign_swap", 100.),
                                        (f"nzd_sofr_tbs_{tenor}_nzbbco{suffix}", "tenor_basis", 1.)]:
            pair = pd.concat([wide[variable].rename("new"), old[source].rename("old")], axis=1).dropna()
            diff = scale * (pair.new - pair.old)
            revisions.append({"variable": variable, "n_overlap": len(pair), "mae_bps": diff.abs().mean(), "max_abs_revision_bps": diff.abs().max()})
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True), pd.DataFrame(revisions)


def plots(coef, increments, nzd_daily):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = ["VIX only", "+ Equity / IG", "+ HY", "+ MOVE / USD"]
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = coef[coef.cohort.eq("fixed8") & coef.tenor.eq(tenor) & coef.window.eq("post2023")
                   & coef.outcome.eq("x_govt_raw_bps") & coef.factor.eq("dlog_vix") & coef.hac_weeks.eq(26)].set_index("spec").loc[list(SPECS)]
        ax.errorbar(labels, sub.beta, yerr=[sub.beta - sub.ci95_low, sub.ci95_high - sub.beta], fmt="o", capsize=4, color="#246b91")
        ax.axhline(0, color="#444444", lw=.8)
        ax.set_title(tenor.upper())
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("VIX loading (bps per log change); 95% HAC-26 interval")
    fig.suptitle("2023 to September 2026: VIX loading on identical weekly dates")
    fig.tight_layout()
    fig.savefig(FIGURE / "credit_risk_vix_conditional.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sub = increments[increments.cohort.eq("fixed8") & increments.outcome.eq("x_govt_raw_bps") & ~increments.window.str.endswith("interaction")]
    windows = ["2010-2015", "2016-2020", "2021-2025", "post2023"]
    for i, (tenor, color) in enumerate([("5y", "#246b91"), ("10y", "#a65b3e")]):
        vals = sub[sub.tenor.eq(tenor)].set_index("window").loc[windows, "delta_r_squared"] * 100
        ax.bar(np.arange(4) + (i - .5) * .35, vals, .35, label=tenor.upper(), color=color)
    ax.set_xticks(np.arange(4), windows)
    ax.set_ylabel("In-sample R-squared increase (percentage points)")
    ax.set_title("Incremental fit from HY, conditional on VIX, equity and IG")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(FIGURE / "credit_risk_hy_incremental_fit.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = nzd_daily[nzd_daily.tenor.eq(tenor)].set_index("date").resample("ME")[["bank_minus_ocr_bps", "basis_bps"]].mean()
        ax.plot(sub.index, sub.bank_minus_ocr_bps, color="#246b91", label="Observed bank swap minus OIS")
        ax.plot(sub.index, sub.basis_bps, color="#a65b3e", ls="--", label="Downloaded tenor basis")
        ax.set_title(tenor.upper())
        ax.set_xticks(pd.to_datetime(["2023-01-31", "2024-01-31", "2025-01-31", "2026-09-30"]), ["2023-01", "2024-01", "2025-01", "2026-09"])
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Basis points; monthly averages of paired daily quotes")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("NZD numerical cross-check; quotes are not assumed to be convention-equivalent")
    fig.tight_layout()
    fig.savefig(FIGURE / "credit_risk_nzd_quote_check.png", dpi=180)
    plt.close(fig)


def main():
    inputs, audits, sources = [], [], []
    for filename in FILES:
        path = SNAPSHOT / filename
        values, audit = read_snapshot(path)
        inputs.append(values)
        audits.append(audit)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    values, audit = pd.concat(inputs, ignore_index=True), pd.concat(audits, ignore_index=True)
    assert len(audit) == 12
    values.to_csv(CLEAN / "credit_risk_downloaded_values.csv", index=False)
    audit.to_csv(TABLE / "credit_risk_download_qc.csv", index=False)
    raw = pd.read_csv(PANEL, parse_dates=["date"], low_memory=False)
    market = values[values.file.eq(FILES[0])].pivot(index="date", columns="variable", values="value")
    weekly = make_weekly(raw, market)
    weekly.to_csv(CLEAN / "credit_risk_weekly_panel.csv", index=False)
    coef, fits, increments = fit_models(weekly)
    coef.to_csv(TABLE / "credit_risk_coefficients.csv", index=False)
    fits.to_csv(TABLE / "credit_risk_model_fits.csv", index=False)
    increments.to_csv(TABLE / "credit_risk_hy_incremental.csv", index=False)
    nz_summary, nz_daily, revisions = nzd_checks(values[values.file.eq(FILES[1])], raw)
    nz_summary.to_csv(TABLE / "credit_risk_nzd_checks.csv", index=False)
    revisions.to_csv(TABLE / "credit_risk_nzd_revisions.csv", index=False)
    plots(coef, increments, nz_daily)
    summary = {"updated": "2026-11-02", "sources": sources, "numeric_series": len(audit), "numeric_quotes": len(values),
               "last_quote": str(values.date.max().date()), "last_complete_risk_week": str(weekly.week.max().date()),
               "regression_fits": len(fits), "coefficient_rows": len(coef),
               "unit_convention": "IG/HY raw percentage points converted x100 to bps; manual unit cells still blank",
               "nzd_status": "OIS/3M names resolved by terminal BDP; detailed DES and pricing conventions remain open",
               "outcome_snapshot": "2026-10-09 CIP snapshot unchanged; new market factors and NZD diagnostics only",
               "caveats": ["Not full/exact paper Table 5 replication: fixed-eight means, matched dates and candidate IG series.",
                           "R-squared changes are in-sample conditional fit, not predictive or causal evidence.",
                           "Corporate OAS embeds a Treasury benchmark and may share mechanical drivers with swap spreads.",
                           "HAC intervals and p-values are exploratory, not multiplicity-adjusted.",
                           "Latest common-regime pre-2023 comparison is short; do not infer a structural break from one specification."]}
    (TABLE / "credit_risk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
