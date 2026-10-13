from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / "output" / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned"
DISCOVERY_DIR = PROJECT_ROOT / "data" / "source_discovery"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"

TARGET_MAP = {
    "x_govt_raw_bps_5y": {
        "label": "5Y",
        "supply": "marketable_maturity_3y_to_6y_mil",
        "dealer": "dealer_coupon_position_3y_to_6y",
    },
    "x_govt_raw_bps_10y": {
        "label": "10Y",
        "supply": "marketable_maturity_7y_to_11y_mil",
        "dealer": "dealer_coupon_position_7y_to_11y",
    },
}

LEVEL_CONTROLS = [
    "bbdxy_index",
    "vix_index",
    "move_index",
    "us_generic_govt_5y_yield",
    "ust_10y_2y_slope",
]
CHANGE_CONTROLS = [f"{col}_change_12m" for col in LEVEL_CONTROLS]
DEALER_CONTROLS = ["dealer_repo_treasury_ex_tips"]


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std()
    if not std or np.isnan(std):
        return x * np.nan
    return (x - x.mean()) / std


def with_time_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_trend"] = (out["date"] - out["date"].min()).dt.days / 365.25
    return out


def residualize_on_trend(data: pd.DataFrame, col: str) -> pd.Series:
    valid = data[["date", col]].dropna().copy()
    valid = with_time_trend(valid)
    model = sm.OLS(valid[col], sm.add_constant(valid[["time_trend"]])).fit()
    out = pd.Series(index=data.index, dtype=float)
    out.loc[valid.index] = model.resid
    return out


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(CLEANED_DIR / "eq5_ust_supply_dealer_official_monthly_panel.csv", parse_dates=["date"])
    return panel.sort_values("date").reset_index(drop=True)


def nyfed_seriesbreaks() -> pd.DataFrame:
    breaks = json.loads((DISCOVERY_DIR / "nyfed_pd_seriesbreaks.json").read_text(encoding="utf-8"))["pd"]["seriesbreaks"]
    break_df = pd.DataFrame(breaks)
    break_df["startdate"] = pd.to_datetime(break_df["startdate"], errors="coerce")
    break_df["enddate"] = pd.to_datetime(break_df["enddate"].replace("9999-12-31", "2100-12-31"), errors="coerce")
    return break_df


def add_seriesbreak_labels(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["nyfed_seriesbreak"] = pd.NA
    for _, row in nyfed_seriesbreaks().iterrows():
        mask = out["date"].between(row["startdate"], row["enddate"])
        out.loc[mask, "nyfed_seriesbreak"] = row["seriesbreak"]
    return out


def correlation_modes(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, spec in TARGET_MAP.items():
        for variable_type, variable in [("same_maturity_supply", spec["supply"]), ("same_maturity_dealer", spec["dealer"])]:
            for sample_name, sample in [
                ("full_available", panel),
                ("post_2013", panel[panel["date"].ge(pd.Timestamp("2013-04-30"))]),
            ]:
                if variable_type == "same_maturity_dealer" and sample_name == "full_available":
                    continue
                level = sample[["date", target, variable]].dropna()
                if len(level) >= 36:
                    rows.append(
                        {
                            "target": target,
                            "variable_type": variable_type,
                            "variable": variable,
                            "sample": sample_name,
                            "mode": "level",
                            "n_months": len(level),
                            "date_min": level["date"].min().date().isoformat(),
                            "date_max": level["date"].max().date().isoformat(),
                            "correlation": level[target].corr(level[variable]),
                        }
                    )
                    detrend = level.copy()
                    detrend[f"{target}_detrended"] = residualize_on_trend(detrend, target)
                    detrend[f"{variable}_detrended"] = residualize_on_trend(detrend, variable)
                    rows.append(
                        {
                            "target": target,
                            "variable_type": variable_type,
                            "variable": variable,
                            "sample": sample_name,
                            "mode": "linear_detrended_level",
                            "n_months": len(detrend),
                            "date_min": detrend["date"].min().date().isoformat(),
                            "date_max": detrend["date"].max().date().isoformat(),
                            "correlation": detrend[f"{target}_detrended"].corr(detrend[f"{variable}_detrended"]),
                        }
                    )

                target_chg = f"{target}_change_12m"
                variable_chg = f"{variable}_change_12m"
                if target_chg in sample.columns and variable_chg in sample.columns:
                    chg = sample[["date", target_chg, variable_chg]].dropna()
                    if len(chg) >= 36:
                        rows.append(
                            {
                                "target": target,
                                "variable_type": variable_type,
                                "variable": variable,
                                "sample": sample_name,
                                "mode": "12m_change",
                                "n_months": len(chg),
                                "date_min": chg["date"].min().date().isoformat(),
                                "date_max": chg["date"].max().date().isoformat(),
                                "correlation": chg[target_chg].corr(chg[variable_chg]),
                            }
                        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "eq5_supply_dealer_robustness_correlation_modes.csv", index=False)
    return out


def run_model(panel: pd.DataFrame, target: str, model_name: str, y_col: str, x_cols: list[str], add_trend: bool) -> tuple[dict, list[dict]]:
    cols = ["date", y_col, *x_cols]
    data = panel[cols].dropna().copy()
    if len(data) < 48:
        return {}, []
    if add_trend:
        data = with_time_trend(data)
        x_cols = [*x_cols, "time_trend"]
    y = pd.to_numeric(data[y_col], errors="coerce")
    x = data[x_cols].copy()
    for col in x.columns:
        if col != "time_trend" and not col.startswith("break_"):
            x[col] = zscore(x[col])
    model_data = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(model_data) < 48:
        return {}, []

    model = sm.OLS(model_data["y"], sm.add_constant(model_data.drop(columns=["y"]))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 6},
    )
    model_row = {
        "target": target,
        "spec": model_name,
        "n_months": int(model.nobs),
        "date_min": data.loc[model_data.index, "date"].min().date().isoformat(),
        "date_max": data.loc[model_data.index, "date"].max().date().isoformat(),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }
    coef_rows = [
        {
            "target": target,
            "spec": model_name,
            "variable": col,
            "coef_bps_per_1sd_or_year": model.params.get(col, np.nan),
            "t_stat_hac": model.tvalues.get(col, np.nan),
            "p_value_hac": model.pvalues.get(col, np.nan),
            "n_months": int(model.nobs),
            "r_squared": model.rsquared,
        }
        for col in x_cols
    ]
    return model_row, coef_rows


def regression_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    coef_rows = []
    for target, spec in TARGET_MAP.items():
        supply = spec["supply"]
        dealer = spec["dealer"]
        full_specs = [
            ("full_supply_level_with_trend", panel, target, [supply], True),
            ("full_supply_controls_level_with_trend", panel, target, [supply, *LEVEL_CONTROLS], True),
            ("full_supply_controls_12m_change", panel, f"{target}_change_12m", [f"{supply}_change_12m", *CHANGE_CONTROLS], False),
        ]
        post_2013 = panel[panel["date"].ge(pd.Timestamp("2013-04-30"))].copy()
        post_specs = [
            (
                "post2013_supply_dealer_controls_level_with_trend",
                post_2013,
                target,
                [supply, dealer, *DEALER_CONTROLS, *LEVEL_CONTROLS],
                True,
            ),
            (
                "post2013_supply_dealer_controls_12m_change",
                post_2013,
                f"{target}_change_12m",
                [f"{supply}_change_12m", f"{dealer}_change_12m", "dealer_repo_treasury_ex_tips_change_12m", *CHANGE_CONTROLS],
                False,
            ),
        ]
        for model_name, sample, y_col, x_cols, add_trend in [*full_specs, *post_specs]:
            x_cols = [col for col in x_cols if col in sample.columns]
            model_row, rows = run_model(sample, target, model_name, y_col, x_cols, add_trend)
            if model_row:
                model_rows.append(model_row)
                coef_rows.extend(rows)

    models = pd.DataFrame(model_rows)
    coefs = pd.DataFrame(coef_rows)
    models.to_csv(TABLE_DIR / "eq5_supply_dealer_robustness_regression_models.csv", index=False)
    coefs.to_csv(TABLE_DIR / "eq5_supply_dealer_robustness_regression_coefficients.csv", index=False)
    return coefs, models


def seriesbreak_coverage() -> pd.DataFrame:
    break_df = nyfed_seriesbreaks()

    weekly = pd.read_csv(CLEANED_DIR / "nyfed_primary_dealer_selected_weekly_long.csv", parse_dates=["date"])
    rows = []
    for variable, group in weekly.groupby("variable"):
        for _, br in break_df.iterrows():
            sub = group[group["date"].between(br["startdate"], br["enddate"])]
            if sub.empty:
                continue
            rows.append(
                {
                    "variable": variable,
                    "seriesbreak": br["seriesbreak"],
                    "label": br["label"],
                    "n_weeks": len(sub),
                    "date_min": sub["date"].min().date().isoformat(),
                    "date_max": sub["date"].max().date().isoformat(),
                    "value_mean": sub["value"].mean(),
                    "value_median": sub["value"].median(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "nyfed_primary_dealer_seriesbreak_coverage.csv", index=False)
    return out


def add_break_dummies(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = sample.copy()
    dummies = pd.get_dummies(data["nyfed_seriesbreak"], prefix="break", dtype=float)
    if "break_SBN2013" in dummies.columns:
        dummies = dummies.drop(columns=["break_SBN2013"])
    data = pd.concat([data, dummies], axis=1)
    return data, dummies.columns.tolist()


def reporting_break_dummy_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = add_seriesbreak_labels(panel)
    post_2013 = panel[panel["date"].ge(pd.Timestamp("2013-04-30"))].copy()
    post_2013, break_cols = add_break_dummies(post_2013)

    model_rows = []
    coef_rows = []
    for target, spec in TARGET_MAP.items():
        supply = spec["supply"]
        dealer = spec["dealer"]
        model_specs = [
            (
                "post2013_level_with_trend_and_break_dummies",
                target,
                [supply, dealer, *DEALER_CONTROLS, *LEVEL_CONTROLS, *break_cols],
                True,
            ),
            (
                "post2013_12m_change_with_break_dummies",
                f"{target}_change_12m",
                [f"{supply}_change_12m", f"{dealer}_change_12m", "dealer_repo_treasury_ex_tips_change_12m", *CHANGE_CONTROLS, *break_cols],
                False,
            ),
        ]
        for model_name, y_col, x_cols, add_trend in model_specs:
            x_cols = [col for col in x_cols if col in post_2013.columns]
            model_row, rows = run_model(post_2013, target, model_name, y_col, x_cols, add_trend)
            if model_row:
                model_rows.append(model_row)
                coef_rows.extend(rows)

    models = pd.DataFrame(model_rows)
    coefs = pd.DataFrame(coef_rows)
    models.to_csv(TABLE_DIR / "eq5_supply_dealer_reporting_break_dummy_models.csv", index=False)
    coefs.to_csv(TABLE_DIR / "eq5_supply_dealer_reporting_break_dummy_coefficients.csv", index=False)
    return coefs, models


def within_break_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    panel = add_seriesbreak_labels(panel)
    rows = []
    for target, spec in TARGET_MAP.items():
        for break_name, sample in panel.dropna(subset=["nyfed_seriesbreak"]).groupby("nyfed_seriesbreak"):
            for variable_type, variable in [("same_maturity_supply", spec["supply"]), ("same_maturity_dealer", spec["dealer"])]:
                level = sample[["date", target, variable]].dropna()
                if len(level) >= 12:
                    rows.append(
                        {
                            "target": target,
                            "seriesbreak": break_name,
                            "variable_type": variable_type,
                            "variable": variable,
                            "mode": "level",
                            "n_months": len(level),
                            "date_min": level["date"].min().date().isoformat(),
                            "date_max": level["date"].max().date().isoformat(),
                            "correlation": level[target].corr(level[variable]),
                        }
                    )
                target_chg = f"{target}_change_12m"
                variable_chg = f"{variable}_change_12m"
                if target_chg in sample.columns and variable_chg in sample.columns:
                    chg = sample[["date", target_chg, variable_chg]].dropna()
                    if len(chg) >= 8:
                        rows.append(
                            {
                                "target": target,
                                "seriesbreak": break_name,
                                "variable_type": variable_type,
                                "variable": variable,
                                "mode": "12m_change",
                                "n_months": len(chg),
                                "date_min": chg["date"].min().date().isoformat(),
                                "date_max": chg["date"].max().date().isoformat(),
                                "correlation": chg[target_chg].corr(chg[variable_chg]),
                            }
                        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "eq5_supply_dealer_within_break_diagnostics.csv", index=False)
    return out


def make_reporting_break_figure(dummy_coefs: pd.DataFrame, within_breaks: pd.DataFrame) -> Path:
    out = FIGURE_DIR / "eq5_supply_dealer_reporting_break_checks.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    selected = dummy_coefs[
        dummy_coefs["variable"].str.contains("marketable_maturity|dealer_coupon", na=False)
        & dummy_coefs["spec"].str.contains("break_dummies", na=False)
    ].copy()
    selected["model_label"] = np.where(selected["spec"].str.contains("12m_change", na=False), "12m change", "level")
    selected["label"] = (
        selected["target"].str.replace("x_govt_raw_bps_", "", regex=False)
        + " "
        + selected["model_label"]
        + "\n"
        + selected["variable"].str.replace("_change_12m", "", regex=False)
    )
    y = np.arange(len(selected))
    axes[0].barh(y, selected["coef_bps_per_1sd_or_year"], color="#4c78a8")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(selected["label"], fontsize=7)
    axes[0].set_title("Post-2013 coefficients with reporting-break dummies")
    axes[0].set_xlabel("bps per 1 SD")

    heat = within_breaks[
        within_breaks["variable_type"].eq("same_maturity_dealer")
        & within_breaks["mode"].eq("12m_change")
    ].pivot_table(index="target", columns="seriesbreak", values="correlation", aggfunc="first")
    im = axes[1].imshow(heat.fillna(0), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_xticks(np.arange(len(heat.columns)))
    axes[1].set_xticklabels(heat.columns, rotation=30, ha="right", fontsize=8)
    axes[1].set_yticks(np.arange(len(heat.index)))
    axes[1].set_yticklabels(heat.index, fontsize=8)
    axes[1].set_title("Within-break dealer 12m-change correlations")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.iloc[i, j]
            if pd.notna(val):
                axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], shrink=0.75)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def make_figure(panel: pd.DataFrame, corr: pd.DataFrame) -> Path:
    out = FIGURE_DIR / "eq5_supply_dealer_robustness_checks.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, target in zip(axes[0], TARGET_MAP):
        supply = TARGET_MAP[target]["supply"]
        data = panel[["date", target, supply]].dropna().copy()
        data[f"{target}_detrended"] = residualize_on_trend(data, target)
        data[f"{supply}_detrended"] = residualize_on_trend(data, supply)
        ax.scatter(zscore(data[f"{supply}_detrended"]), data[f"{target}_detrended"], s=18, alpha=0.6)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(f"{TARGET_MAP[target]['label']} detrended: supply vs UST convenience")
        ax.set_xlabel("detrended supply z-score")
        ax.set_ylabel("detrended UST convenience residual, bps")

    for ax, target in zip(axes[1], TARGET_MAP):
        supply = TARGET_MAP[target]["supply"]
        dealer = TARGET_MAP[target]["dealer"]
        sub = panel[panel["date"].ge(pd.Timestamp("2013-04-30"))].copy()
        ax.plot(sub["date"], zscore(sub[f"{target}_change_12m"]), label="UST convenience 12m change", lw=1.0)
        ax.plot(sub["date"], zscore(sub[f"{supply}_change_12m"]), label="same-maturity supply 12m change", lw=1.0)
        ax.plot(sub["date"], zscore(sub[f"{dealer}_change_12m"]), label="same-maturity dealer position 12m change", lw=1.0, alpha=0.8)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"{TARGET_MAP[target]['label']} post-2013 12m-change diagnostics")
        ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_memo(
    corr: pd.DataFrame,
    coefs: pd.DataFrame,
    models: pd.DataFrame,
    breaks: pd.DataFrame,
    dummy_coefs: pd.DataFrame,
    dummy_models: pd.DataFrame,
    within_breaks: pd.DataFrame,
    figure: Path,
    break_figure: Path,
) -> Path:
    def corr_pick(target: str, variable_type: str, sample: str, mode: str) -> float:
        rows = corr[
            corr["target"].eq(target)
            & corr["variable_type"].eq(variable_type)
            & corr["sample"].eq(sample)
            & corr["mode"].eq(mode)
        ]
        return float(rows.iloc[0]["correlation"]) if not rows.empty else np.nan

    def coef_pick(target: str, spec: str, contains: str) -> pd.Series | None:
        rows = coefs[coefs["target"].eq(target) & coefs["spec"].eq(spec) & coefs["variable"].str.contains(contains, na=False)]
        return rows.iloc[0] if not rows.empty else None

    robust_5y = corr_pick("x_govt_raw_bps_5y", "same_maturity_supply", "full_available", "linear_detrended_level")
    robust_10y = corr_pick("x_govt_raw_bps_10y", "same_maturity_supply", "full_available", "linear_detrended_level")
    chg_5y = corr_pick("x_govt_raw_bps_5y", "same_maturity_supply", "full_available", "12m_change")
    chg_10y = corr_pick("x_govt_raw_bps_10y", "same_maturity_supply", "full_available", "12m_change")

    beta_5y = coef_pick("x_govt_raw_bps_5y", "post2013_supply_dealer_controls_level_with_trend", "marketable_maturity")
    beta_10y = coef_pick("x_govt_raw_bps_10y", "post2013_supply_dealer_controls_level_with_trend", "marketable_maturity")
    beta_10y_dealer = coef_pick("x_govt_raw_bps_10y", "post2013_supply_dealer_controls_level_with_trend", "dealer_coupon")
    full_5y_supply = coef_pick("x_govt_raw_bps_5y", "full_supply_controls_level_with_trend", "marketable_maturity")
    full_10y_supply = coef_pick("x_govt_raw_bps_10y", "full_supply_controls_level_with_trend", "marketable_maturity")
    chg_5y_supply = coef_pick("x_govt_raw_bps_5y", "full_supply_controls_12m_change", "marketable_maturity")
    chg_10y_dealer = coef_pick("x_govt_raw_bps_10y", "post2013_supply_dealer_controls_12m_change", "dealer_coupon")

    def dummy_coef_pick(target: str, spec: str, contains: str) -> pd.Series | None:
        rows = dummy_coefs[
            dummy_coefs["target"].eq(target)
            & dummy_coefs["spec"].eq(spec)
            & dummy_coefs["variable"].str.contains(contains, na=False)
        ]
        return rows.iloc[0] if not rows.empty else None

    dummy_10y_dealer_change = dummy_coef_pick(
        "x_govt_raw_bps_10y",
        "post2013_12m_change_with_break_dummies",
        "dealer_coupon",
    )
    dummy_5y_supply_change = dummy_coef_pick(
        "x_govt_raw_bps_5y",
        "post2013_12m_change_with_break_dummies",
        "marketable_maturity",
    )
    within_10y_dealer = within_breaks[
        within_breaks["target"].eq("x_govt_raw_bps_10y")
        & within_breaks["variable_type"].eq("same_maturity_dealer")
        & within_breaks["mode"].eq("12m_change")
    ].sort_values("seriesbreak")

    break_labels = ", ".join(breaks["seriesbreak"].drop_duplicates().tolist())
    lines = [
        "# Eq.5 Supply / Dealer Robustness Checks",
        "",
        f"Generated: {pd.Timestamp.today().date().isoformat()}",
        "",
        "## Purpose",
        "",
        "This checks whether the official supply/dealer first-pass result survives more conservative specifications.",
        "The main concern is that Treasury convenience and Treasury supply are both persistent level series, so raw level correlations may overstate the economic relationship.",
        "",
        "## Main Results",
        "",
        f"- Full-sample linear-detrended correlation, 5Y convenience vs 3-6Y supply: {robust_5y:.3f}.",
        f"- Full-sample linear-detrended correlation, 10Y convenience vs 7-11Y supply: {robust_10y:.3f}.",
        f"- Full-sample 12-month-change correlation, 5Y convenience vs 3-6Y supply: {chg_5y:.3f}.",
        f"- Full-sample 12-month-change correlation, 10Y convenience vs 7-11Y supply: {chg_10y:.3f}.",
        "",
        "Selected full-sample coefficients:",
    ]
    for label, row in [
        ("5Y same-maturity supply, controls + trend", full_5y_supply),
        ("10Y same-maturity supply, controls + trend", full_10y_supply),
        ("5Y same-maturity supply, 12-month changes + controls", chg_5y_supply),
    ]:
        if row is not None:
            lines.append(
                f"- {label}: {row['coef_bps_per_1sd_or_year']:.2f} bps per 1 SD, HAC t={row['t_stat_hac']:.2f}."
            )

    lines.extend(
        [
            "",
            "Selected post-2013 coefficients:",
        ]
    )
    for label, row in [
        ("5Y same-maturity supply, level controls + trend", beta_5y),
        ("10Y same-maturity supply, level controls + trend", beta_10y),
        ("10Y same-maturity dealer position, level controls + trend", beta_10y_dealer),
        ("10Y same-maturity dealer position, 12-month changes + controls", chg_10y_dealer),
    ]:
        if row is not None:
            lines.append(
                f"- {label}: {row['coef_bps_per_1sd_or_year']:.2f} bps per 1 SD, HAC t={row['t_stat_hac']:.2f}."
            )
    lines.extend(
        [
            "",
            "Selected post-2013 coefficients after adding reporting-break dummies:",
        ]
    )
    for label, row in [
        ("5Y same-maturity supply, 12-month changes + controls + break dummies", dummy_5y_supply_change),
        ("10Y same-maturity dealer position, 12-month changes + controls + break dummies", dummy_10y_dealer_change),
    ]:
        if row is not None:
            lines.append(
                f"- {label}: {row['coef_bps_per_1sd_or_year']:.2f} bps per 1 SD, HAC t={row['t_stat_hac']:.2f}."
            )

    if not within_10y_dealer.empty:
        lines.extend(["", "Within-break 10Y dealer-position 12-month-change correlations:"])
        for _, row in within_10y_dealer.iterrows():
            lines.append(f"- {row['seriesbreak']}: {row['correlation']:.3f} over {int(row['n_months'])} months.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The raw level relationship between Treasury supply and UST convenience is strong, but detrending and 12-month changes weaken it materially.",
            "- The full-sample 5Y supply signal survives better than the post-2013 level-with-dealer specification.",
            "- In the post-2013 dealer sample, same-maturity level coefficients weaken; reporting-break dummies are necessary before taking the dealer coefficients seriously.",
            "- This means the evidence should be framed as a plausible supply/intermediation channel, not a causal result yet.",
            "- Dealer results should be treated carefully because the selected NY Fed series cross reporting breaks.",
            "",
            "## NY Fed Series-Break Note",
            "",
            f"The selected NY Fed dealer series cover these reporting regimes in the local data: {break_labels}.",
            "Before writing a formal result, we should either include reporting-break dummies, use within-break robustness checks, or document why the current continuous series is acceptable.",
            "",
            "## Outputs",
            "",
            "- Correlation modes: `output/tables/eq5_supply_dealer_robustness_correlation_modes.csv`",
            "- Regression models: `output/tables/eq5_supply_dealer_robustness_regression_models.csv`",
            "- Regression coefficients: `output/tables/eq5_supply_dealer_robustness_regression_coefficients.csv`",
            "- NY Fed series-break coverage: `output/tables/nyfed_primary_dealer_seriesbreak_coverage.csv`",
            "- Reporting-break dummy models: `output/tables/eq5_supply_dealer_reporting_break_dummy_models.csv`",
            "- Reporting-break dummy coefficients: `output/tables/eq5_supply_dealer_reporting_break_dummy_coefficients.csv`",
            "- Within-break diagnostics: `output/tables/eq5_supply_dealer_within_break_diagnostics.csv`",
            f"- Figure: `{figure.relative_to(PROJECT_ROOT)}`",
            f"- Reporting-break figure: `{break_figure.relative_to(PROJECT_ROOT)}`",
        ]
    )
    out = REPORT_DIR / "Eq5_Supply_Dealer_Robustness_Memo.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    corr = correlation_modes(panel)
    coefs, models = regression_tables(panel)
    breaks = seriesbreak_coverage()
    dummy_coefs, dummy_models = reporting_break_dummy_tables(panel)
    within_breaks = within_break_diagnostics(panel)
    figure = make_figure(panel, corr)
    break_figure = make_reporting_break_figure(dummy_coefs, within_breaks)
    memo = write_memo(corr, coefs, models, breaks, dummy_coefs, dummy_models, within_breaks, figure, break_figure)

    for path in [
        TABLE_DIR / "eq5_supply_dealer_robustness_correlation_modes.csv",
        TABLE_DIR / "eq5_supply_dealer_robustness_regression_models.csv",
        TABLE_DIR / "eq5_supply_dealer_robustness_regression_coefficients.csv",
        TABLE_DIR / "nyfed_primary_dealer_seriesbreak_coverage.csv",
        TABLE_DIR / "eq5_supply_dealer_reporting_break_dummy_models.csv",
        TABLE_DIR / "eq5_supply_dealer_reporting_break_dummy_coefficients.csv",
        TABLE_DIR / "eq5_supply_dealer_within_break_diagnostics.csv",
        figure,
        break_figure,
        memo,
    ]:
        print(path)


if __name__ == "__main__":
    main()
