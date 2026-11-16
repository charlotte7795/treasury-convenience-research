"""Paper-aligned risk checks, joint Wald tests, and fixed-rule stress weeks."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

from src.analyze_credit_risk_extension import FACTORS, OUTCOMES, SPECS
from src.analyze_post_meeting_research import weekly_hac
from src.analyze_treasury_interim import CLEAN, COHORT, FIGURE, ROOT, TABLE, plt
from src.refresh_treasury_5y10y import PANEL

PERIODS = {"2010-2015": ("2010-01-01", "2015-12-31"),
           "2016-2020": ("2016-01-01", "2020-12-31"),
           "2021-2025H1": ("2021-01-01", "2025-06-30")}
PAPER_GOVT = {
    "5y": {"dlog_vix": [6.37, -.02, -5.50], "spx_return": [-52.51, .95, 38.45], "d_ig_bps": [20.24, -2.81, -11.17]},
    "10y": {"dlog_vix": [4.69, -1.10, -4.54], "spx_return": [-47.41, 12.21, 28.91], "d_ig_bps": [19.20, -5.56, -8.46]},
}
STRESS_QUANTILE = .90
BOOTSTRAP_REPS = 2000
BOOTSTRAP_BLOCK = 8
BOOTSTRAP_SEED = 20261102


def paper_comparison(raw, public, market, weekly):
    rows = []
    g10 = [*COHORT, "JPY", "NOK"]
    for tenor in ["5y", "10y"]:
        sub = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT) & raw.tenor.eq(tenor)]
        wide = sub.pivot(index="date", columns="currency", values="x_govt_raw_bps").reindex(columns=COHORT)
        dates = wide.dropna().index.intersection(market.dropna().index)
        dates = dates[(dates >= "2010-01-01") & (dates <= "2025-06-27")]
        p = public[public.currency.isin(g10) & public.tenor.eq(tenor)].pivot(index="date", columns="currency", values="cip_govt").reindex(index=dates, columns=g10)
        assert p.count(axis=1).gt(0).all()
        daily = pd.DataFrame({"public_G10_available_mean": p.mean(axis=1), "public_G10_available_median": p.median(axis=1)})
        aggregate = daily.resample("W-FRI").mean()
        differences = aggregate.diff().add_prefix("d_")
        sample = weekly[weekly.cohort.eq("fixed8") & weekly.tenor.eq(tenor)].set_index("week")
        sample = sample.join(differences, how="left", validate="one_to_one")
        variants = {"raw_fixed8_mean": "d_x_govt_raw_bps", **{c: f"d_{c}" for c in daily.columns}}
        sample = sample[sample.valid_change].dropna(subset=list(variants.values()) + FACTORS[:3])
        for period_index, (period, bounds) in enumerate(PERIODS.items()):
            window = sample.loc[bounds[0]:bounds[1]]
            for factor in FACTORS[:3]:
                x = sm.add_constant(window[[factor]])
                for variant, outcome in variants.items():
                    model = sm.OLS(window[outcome], x).fit()
                    se = np.sqrt(weekly_hac(model, window.index)[1, 1])
                    beta = model.params[factor]
                    # The paper's IG unit is not explicit: show per-pp only as a candidate scale.
                    scale = 100. if factor == "d_ig_bps" else 1.
                    rows.append({"variant": variant, "tenor": tenor, "period": period, "factor": factor,
                                 "n_weeks": len(window), "first_week": window.index.min(), "last_week": window.index.max(),
                                 "raw_beta": beta, "beta_on_display_scale": beta * scale,
                                 "ci95_low": (beta - 1.96 * se) * scale, "ci95_high": (beta + 1.96 * se) * scale,
                                 "paper_printed_beta": PAPER_GOVT[tenor][factor][period_index],
                                 "same_sign": bool(np.sign(beta) == np.sign(PAPER_GOVT[tenor][factor][period_index])),
                                 "scale_status": "IG per percentage point candidate; exact paper scale unresolved" if factor == "d_ig_bps" else "bps per log change",
                                 "status": "Directional partial replication; no exact-match claim"})
    return pd.DataFrame(rows)


def joint_wald(beta, covariance, restriction):
    contrast = restriction @ np.asarray(beta)
    variance = restriction @ covariance @ restriction.T
    assert np.linalg.matrix_rank(variance) == len(restriction)
    statistic = float(contrast @ np.linalg.solve(variance, contrast))
    return statistic, float(chi2.sf(statistic, len(restriction)))


def joint_tests(weekly):
    rows = []
    for (cohort, tenor), sub in weekly.groupby(["cohort", "tenor"]):
        sub = sub[sub.valid_change].set_index("week").dropna(subset=FACTORS + [f"d_{m}" for m in OUTCOMES])
        windows = {"post2023": sub.index >= "2023-01-01", "full_interaction": np.ones(len(sub), dtype=bool),
                   "recent_regime_interaction": sub.signature.eq(sub.signature.iloc[-1]).to_numpy()}
        for window, mask in windows.items():
            sample = sub.loc[mask]
            interaction = window.endswith("interaction")
            for spec in ["VIX_equity_IG", "plus_HY"]:
                factors = SPECS[spec]
                x = sample[factors].copy()
                if interaction:
                    x["post2023"] = (sample.index >= "2023-01-01").astype(float)
                    for factor in factors:
                        x[f"{factor}_post2023"] = x[factor] * x.post2023
                    if window == "full_interaction":
                        x = pd.concat([x, pd.get_dummies(sample.signature, prefix="regime", drop_first=True, dtype=float)], axis=1)
                x = sm.add_constant(x).astype(float)
                assert np.linalg.matrix_rank(x) == x.shape[1]
                tests = ["post_loadings_zero", "slope_changes_zero"] if interaction else ["post_loadings_zero"]
                for outcome in OUTCOMES:
                    model = sm.OLS(sample[f"d_{outcome}"], x).fit()
                    for lag in [13, 26]:
                        covariance = weekly_hac(model, sample.index, lag)
                        for test in tests:
                            restriction = np.zeros((len(factors), len(x.columns)))
                            for i, factor in enumerate(factors):
                                if test == "post_loadings_zero":
                                    restriction[i, x.columns.get_loc(factor)] = 1.
                                if interaction:
                                    restriction[i, x.columns.get_loc(f"{factor}_post2023")] = 1.
                            statistic, p = joint_wald(model.params, covariance, restriction)
                            rows.append({"cohort": cohort, "tenor": tenor, "window": window, "spec": spec,
                                         "outcome": outcome, "test": test, "hac_weeks": lag, "df": len(factors),
                                         "wald_chi2": statistic, "p_exploratory": p, "n_weeks": len(sample),
                                         "n_pre2023": int((sample.index < "2023-01-01").sum()),
                                         "n_post2023": int((sample.index >= "2023-01-01").sum())})
    result = pd.DataFrame(rows)
    primary = (result.cohort.eq("fixed8") & result.spec.eq("plus_HY") & result.outcome.eq("x_govt_raw_bps")
               & result.hac_weeks.eq(26) & ((result.window.eq("post2023") & result.test.eq("post_loadings_zero"))
                                          | (result.window.str.endswith("interaction") & result.test.eq("slope_changes_zero"))))
    assert primary.sum() == 6
    result["p_holm_primary_family"] = np.nan
    result.loc[primary, "p_holm_primary_family"] = multipletests(result.loc[primary, "p_exploratory"], method="holm")[1]
    return result


def stress_rule(vix):
    assert vix.index.is_unique and vix.gt(0).all()
    cutoff = min(vix.index.max(), pd.Timestamp("2026-09-30"))
    cutoff -= pd.Timedelta(days=(cutoff.weekday() - 4) % 7)
    observed = vix.loc[:cutoff].resample("W-FRI")
    levels = observed.mean().where(observed.count().ge(3))
    shocks = np.log(levels).diff().rename("stress_dlog_vix")
    training = shocks.loc["2010-01-01":"2022-12-31"].dropna()
    threshold = float(training.quantile(STRESS_QUANTILE))
    flags = shocks.to_frame()
    flags["stress"] = shocks.gt(threshold).astype(float).where(shocks.notna())
    receipt = {"calibration_start": str(training.index.min().date()), "calibration_end": str(training.index.max().date()),
               "calibration_weeks": len(training), "quantile": STRESS_QUANTILE, "threshold_dlog_vix": threshold,
               "equivalent_vix_percent_increase": 100 * np.expm1(threshold),
               "threshold_inputs": "VIX-only calendar; no Treasury outcomes or post-2022 observations",
               "bootstrap_block_weeks": BOOTSTRAP_BLOCK, "bootstrap_reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED}
    return flags, receipt


def block_bootstrap(calendar, reps=BOOTSTRAP_REPS, block=BOOTSTRAP_BLOCK, seed=BOOTSTRAP_SEED):
    dates = pd.DatetimeIndex(calendar.index)
    assert dates.equals(pd.date_range(dates.min(), dates.max(), freq="W-FRI")), "Keep missing weeks on the calendar"
    data = calendar[["stress", *[f"d_{m}" for m in OUTCOMES]]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        starts = rng.integers(0, len(data), size=int(np.ceil(len(data) / block)))
        positions = ((starts[:, None] + np.arange(block)) % len(data)).ravel()[:len(data)]
        draw = data[positions]
        draw = draw[np.isfinite(draw).all(axis=1)]
        high, normal = draw[draw[:, 0] == 1, 1:], draw[draw[:, 0] == 0, 1:]
        if not len(high) or not len(normal):
            continue
        samples.append(np.stack([high.mean(axis=0), normal.mean(axis=0), high.mean(axis=0) - normal.mean(axis=0)]))
    assert len(samples) >= .95 * reps, "Too few bootstrap samples contain both stress states"
    samples = np.asarray(samples)
    assert np.max(np.abs(samples[:, :, 0] - samples[:, :, 1] - samples[:, :, 2] + samples[:, :, 3])) < 1e-8
    return np.quantile(samples, [.025, .975], axis=0), len(samples)


def stress_tests(weekly, flags):
    rows = []
    periods = {"2010-2015": ("2010-01-01", "2015-12-31"), "2016-2020": ("2016-01-01", "2020-12-31"),
               "2021-2022": ("2021-01-01", "2022-12-31"), "post2023": ("2023-01-01", "2026-09-30")}
    for (cohort, tenor), group in weekly.groupby(["cohort", "tenor"]):
        calendar = group.set_index("week").join(flags, how="left", validate="one_to_one")
        calendar.loc[~calendar.valid_change, [f"d_{m}" for m in OUTCOMES]] = np.nan
        for period, bounds in periods.items():
            window = calendar.loc[bounds[0]:bounds[1]]
            sample = window.dropna(subset=["stress", *[f"d_{m}" for m in OUTCOMES]])
            assert sample.stress.eq(1).sum() >= 3 and sample.stress.eq(0).sum() >= 20
            bootstrap, n_boot = block_bootstrap(window) if period == "post2023" else (None, 0)
            x = sm.add_constant(sample[["stress"]])
            for i, outcome in enumerate(OUTCOMES):
                model = sm.OLS(sample[f"d_{outcome}"], x).fit()
                covariance = weekly_hac(model, sample.index)
                high = sample.loc[sample.stress.eq(1), f"d_{outcome}"]
                ordinary = sample.loc[sample.stress.eq(0), f"d_{outcome}"].mean()
                leave_one = (high.sum() - high) / (len(high) - 1) - ordinary
                for j, (contrast, vector) in enumerate([("stress_mean", [1., 1.]), ("ordinary_mean", [1., 0.]),
                                                        ("stress_minus_ordinary", [0., 1.])]):
                    vector = np.array(vector)
                    estimate = float(vector @ model.params)
                    se = float(np.sqrt(vector @ covariance @ vector))
                    rows.append({"cohort": cohort, "tenor": tenor, "period": period, "outcome": outcome, "contrast": contrast,
                                 "estimate_bps": estimate, "hac26_ci_low": estimate - 1.96 * se, "hac26_ci_high": estimate + 1.96 * se,
                                 "p_exploratory": 2 * norm.sf(abs(estimate / se)), "n_stress": int(sample.stress.eq(1).sum()),
                                 "n_ordinary": int(sample.stress.eq(0).sum()), "n_calendar_weeks": len(window),
                                 "bootstrap_ci_low": bootstrap[0, j, i] if bootstrap is not None else np.nan,
                                 "bootstrap_ci_high": bootstrap[1, j, i] if bootstrap is not None else np.nan,
                                 "bootstrap_valid_reps": n_boot,
                                 "leave_one_stress_out_min": leave_one.min() if contrast == "stress_minus_ordinary" else np.nan,
                                 "leave_one_stress_out_max": leave_one.max() if contrast == "stress_minus_ordinary" else np.nan})
    result = pd.DataFrame(rows)
    bridge = result.pivot(index=["cohort", "tenor", "period", "contrast"], columns="outcome", values="estimate_bps")
    assert (bridge.x_govt_raw_bps - bridge.x_rf_raw_bps - bridge.us_swap_spread_bps + bridge.foreign_swap_spread_bps).abs().max() < 1e-8
    primary = (result.cohort.eq("fixed8") & result.period.eq("post2023") & result.outcome.eq("x_govt_raw_bps")
               & result.contrast.eq("stress_minus_ordinary"))
    assert primary.sum() == 2
    result["p_holm_stress_primary_family"] = np.nan
    result.loc[primary, "p_holm_stress_primary_family"] = multipletests(result.loc[primary, "p_exploratory"], method="holm")[1]
    return result


def make_figures(comparison, stress):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    styles = [("raw_fixed8_mean", "Raw fixed-eight mean", "#246b91", "o"),
              ("public_G10_available_mean", "Public G10 available mean", "#a65b3e", "s"),
              ("public_G10_available_median", "Public G10 available median", "#426b50", "^")]
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = comparison[comparison.tenor.eq(tenor) & comparison.factor.eq("dlog_vix")]
        paper = sub.drop_duplicates("period").set_index("period").loc[list(PERIODS)]
        ax.plot(np.arange(3), paper.paper_printed_beta, "k--x", label="Paper Table 5")
        for variant, label, color, marker in styles:
            data = sub[sub.variant.eq(variant)].set_index("period").loc[list(PERIODS)]
            ax.plot(np.arange(3), data.beta_on_display_scale, color=color, marker=marker, label=label)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_xticks(np.arange(3), list(PERIODS))
        ax.set_title(tenor.upper())
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Univariate VIX beta: bps per log-VIX change")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Paper-aligned date windows; directional comparison, not exact replication")
    fig.tight_layout()
    fig.savefig(FIGURE / "risk_properties_table5_vix_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = stress[stress.cohort.eq("fixed8") & stress.tenor.eq(tenor) & stress.outcome.eq("x_govt_raw_bps")
                     & stress.contrast.eq("stress_minus_ordinary")]
        ax.errorbar(sub.period, sub.estimate_bps, yerr=[sub.estimate_bps - sub.hac26_ci_low, sub.hac26_ci_high - sub.estimate_bps],
                    color="#246b91", fmt="o", capsize=4)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_title(tenor.upper())
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Weekly convenience change: stress minus ordinary (bps)")
    fig.suptitle("Fixed pre-2023 VIX threshold; 95% calendar-HAC intervals")
    fig.tight_layout()
    fig.savefig(FIGURE / "risk_properties_stress_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    labels = ["Treasury convenience (net)", "Dollar-convenience proxy", "U.S. swap spread", "Minus foreign swap spread"]
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = stress[stress.cohort.eq("fixed8") & stress.tenor.eq(tenor) & stress.period.eq("post2023")
                     & stress.contrast.eq("stress_minus_ordinary")].set_index("outcome").loc[OUTCOMES]
        signs = np.array([1., 1., 1., -1.])
        points = sub.estimate_bps.to_numpy() * signs
        ends = np.column_stack([sub.hac26_ci_low, sub.hac26_ci_high]) * signs[:, None]
        ax.errorbar(points, np.arange(4), xerr=[points - ends.min(axis=1), ends.max(axis=1) - points],
                    color="#246b91", fmt="o", capsize=4)
        ax.axvline(0, color="#555555", lw=.8)
        ax.set_yticks(np.arange(4), labels)
        ax.set_ylim(3.5, -.5)
        ax.set_title(tenor.upper())
        ax.set_xlabel("Stress minus ordinary weekly change (bps)")
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("2023 to September 2026: net convenience and additive components\n13 observed stress weeks per tenor; 95% calendar-HAC intervals")
    fig.tight_layout()
    fig.savefig(FIGURE / "risk_properties_stress_components.png", dpi=180)
    plt.close(fig)


def main():
    weekly = pd.read_csv(CLEAN / "credit_risk_weekly_panel.csv", parse_dates=["week"])
    quotes = pd.read_csv(CLEAN / "credit_risk_downloaded_values.csv", parse_dates=["date"])
    market = quotes[quotes.file.eq("Risk_Extension_Bloomberg_BDH_Template.xlsx")].pivot(index="date", columns="variable", values="value")
    raw = pd.read_csv(PANEL, parse_dates=["date"], low_memory=False)
    public = pd.read_csv(CLEAN / "public_cip_cleaned.csv", parse_dates=["date"])
    comparison = paper_comparison(raw, public, market, weekly)
    comparison.to_csv(TABLE / "risk_properties_table5_comparison.csv", index=False)
    joint = joint_tests(weekly)
    joint.to_csv(TABLE / "risk_properties_joint_tests.csv", index=False)
    flags, receipt = stress_rule(market.vix_index.dropna())
    flags.to_csv(CLEAN / "risk_properties_stress_calendar.csv")
    stress = stress_tests(weekly, flags)
    stress.to_csv(TABLE / "risk_properties_stress_results.csv", index=False)
    make_figures(comparison, stress)
    matrix = [
        ["Model", "Equation 7 / Table 5: one risk factor at a time", "Univariate comparison added; prior multi-factor models remain extensions", "Aligned model form"],
        ["Periods", "2010-15, 2016-20, 2021-25; paper data end June 2025", "Final comparison ends 2025-06-27 complete week", "Aligned dates; first change starts Jan 15, 2010 due available history"],
        ["Frequency / inference", "Weekly changes; Newey-West 26 weeks", "Weekly means then changes; calendar-aware HAC 26", "Partly aligned; within-week timing and exact original code unavailable"],
        ["G10 aggregation", "G10 in Table 4/5; exact cross-country weighting is not specified in the inspected text", "Raw fixed-eight mean plus public available-G10 mean/median sensitivities", "Not claimed exact"],
        ["IG risk series", "U.S. investment-grade corporate spread; exact ticker/unit not provided in Table 5", "LUACOAS candidate; per-pp display for comparison only", "Definition / scale unresolved"],
        ["Measure vintage", "Frozen 71-page paper; source measurement described in Section 3", "Existing public V4 and validated raw reconstruction", "Public appendix warns released data differ from paper data"],
        ["Coverage", "Table 5 includes additional tenors, dollar and all components", "18 government-CIP cells at 5Y/10Y only", "Partial replication"],
    ]
    pd.DataFrame(matrix, columns=["dimension", "paper_evidence", "implementation", "remaining_gap"]).to_csv(TABLE / "risk_properties_table5_specification.csv", index=False)
    paper = ROOT / "Decoupling Dollar and Treasury Privilege.pdf"
    receipt.update({"updated": "2026-11-02",
                    "paper_sha256": hashlib.sha256(paper.read_bytes()).hexdigest() if paper.exists() else None,
                    "paper_reference_url": "https://www.federalreserve.gov/econres/ifdp/files/ifdp1427.pdf",
                    "paper_pages_inspected": [13, 23, 24, 25, 50, 51], "table5_rows": len(comparison),
                    "joint_test_rows": len(joint), "stress_result_rows": len(stress),
                    "input_receipts": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                       for path in [PANEL, CLEAN / "public_cip_cleaned.csv", CLEAN / "credit_risk_downloaded_values.csv", CLEAN / "credit_risk_weekly_panel.csv"]],
                    "caveats": ["Threshold fixed before this calculation; not an independent prospective preregistration.",
                                "Bootstrap conditions on the estimated historical threshold; blocks preserve calendar gaps and components.",
                                "Stress association and joint significance do not establish causality or absolute safety.",
                                "Holm adjustments cover six primary government-CIP joint tests and, separately, two primary stress contrasts; not the whole project.",
                                "Leave-one-stress-week-out ranges are influence diagnostics, not confidence intervals."]})
    (TABLE / "risk_properties_summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
