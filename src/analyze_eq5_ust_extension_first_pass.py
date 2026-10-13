from __future__ import annotations

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
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"

CONTROL_VARS = [
    "move_index",
    "vix_index",
    "bbdxy_index",
    "us_generic_govt_5y_yield",
    "us_generic_govt_10y_yield",
    "ust_10y_2y_slope",
    "usgg5yr_bid_ask_spread_bps",
    "usgg10yr_bid_ask_spread_bps",
]


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std()
    if not std or np.isnan(std):
        return x * np.nan
    return (x - x.mean()) / std


def load_eq5_daily() -> pd.DataFrame:
    panel = pd.read_csv(
        CLEANED_DIR / "raw_g10_benchmark_eq6_5y10y_panel.csv",
        parse_dates=["date"],
        low_memory=False,
    )
    panel = panel[panel["is_selected_regime_by_public_break"].eq(True)].copy()
    panel["diff_y_raw_bps"] = 100 * (panel["foreign_govt_yield"] - panel["usd_govt_yield"])
    metrics = [
        "x_govt_raw_bps",
        "diff_y_raw_bps",
        "rho_raw_bps",
        "x_rf_raw_bps",
        "us_swap_spread_bps",
        "foreign_swap_spread_bps",
    ]
    daily = panel.groupby(["date", "tenor"], as_index=False)[metrics].median()
    wide = daily.pivot(index="date", columns="tenor", values=metrics)
    wide.columns = [f"{metric}_{tenor}" for metric, tenor in wide.columns]
    return wide.reset_index().sort_values("date")


def load_controls_daily() -> pd.DataFrame:
    values = pd.read_csv(
        CLEANED_DIR / "downloaded_bloomberg_values_long.csv",
        usecols=["workbook", "date", "variable", "value"],
        parse_dates=["date"],
    )
    values = values[values["workbook"].isin(["macro_controls", "treasury_liquidity_dealer"])].copy()
    wide = values.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    wide.columns.name = None
    for tenor in ["2yr", "5yr", "10yr", "30yr"]:
        bid = f"usgg{tenor}_px_bid"
        ask = f"usgg{tenor}_px_ask"
        if bid in wide.columns and ask in wide.columns:
            wide[f"usgg{tenor}_bid_ask_spread_bps"] = (wide[ask] - wide[bid]).abs() * 100.0
    if {"us_generic_govt_10y_yield", "us_generic_govt_2y_yield"}.issubset(wide.columns):
        wide["ust_10y_2y_slope"] = wide["us_generic_govt_10y_yield"] - wide["us_generic_govt_2y_yield"]
    return wide.reset_index().sort_values("date")


def build_monthly_panel() -> pd.DataFrame:
    eq5 = load_eq5_daily()
    controls = load_controls_daily()
    daily = eq5.merge(controls, on="date", how="left")
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = daily.groupby("month", as_index=False).mean(numeric_only=True)
    monthly = monthly.rename(columns={"month": "date"})
    return monthly


def correlation_table(monthly: pd.DataFrame) -> pd.DataFrame:
    targets = ["x_govt_raw_bps_5y", "x_govt_raw_bps_10y"]
    rows = []
    for target in targets:
        for control in CONTROL_VARS:
            if target not in monthly.columns or control not in monthly.columns:
                continue
            sub = monthly[["date", target, control]].dropna()
            if len(sub) < 36:
                continue
            changes = sub[[target, control]].diff().dropna()
            rows.append(
                {
                    "target": target,
                    "control": control,
                    "n_months": len(sub),
                    "date_min": sub["date"].min().date().isoformat(),
                    "date_max": sub["date"].max().date().isoformat(),
                    "level_corr": sub[target].corr(sub[control]),
                    "change_corr": changes[target].corr(changes[control]),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "level_corr"], key=lambda s: s.abs() if s.name == "level_corr" else s, ascending=False)


def run_regression(y: pd.Series, x: pd.DataFrame, maxlags: int = 6):
    x_std = x.apply(zscore)
    y = pd.to_numeric(y, errors="coerce")
    data = pd.concat([y.rename("y"), x_std], axis=1).dropna()
    if len(data) < 48:
        return None, data
    X = sm.add_constant(data.drop(columns=["y"]))
    model = sm.OLS(data["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return model, data


def regression_tables(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "volatility": ["move_index", "vix_index"],
        "dollar_rates": ["bbdxy_index", "us_generic_govt_5y_yield", "ust_10y_2y_slope"],
        "liquidity_candidates": ["move_index", "usgg5yr_bid_ask_spread_bps", "usgg10yr_bid_ask_spread_bps"],
        "combined": ["move_index", "vix_index", "bbdxy_index", "us_generic_govt_5y_yield", "ust_10y_2y_slope", "usgg5yr_bid_ask_spread_bps"],
    }
    rows = []
    model_summaries = []
    for target in ["x_govt_raw_bps_5y", "x_govt_raw_bps_10y"]:
        for transform in ["level", "change"]:
            for spec_name, vars_ in specs.items():
                available = [v for v in vars_ if v in monthly.columns]
                if transform == "level":
                    y = monthly[target]
                    x = monthly[available]
                else:
                    y = monthly[target].diff()
                    x = monthly[available].diff()
                model, data = run_regression(y, x)
                if model is None:
                    continue
                model_summaries.append(
                    {
                        "target": target,
                        "transform": transform,
                        "spec": spec_name,
                        "n_months": int(model.nobs),
                        "date_min": monthly.loc[data.index, "date"].min().date().isoformat(),
                        "date_max": monthly.loc[data.index, "date"].max().date().isoformat(),
                        "r_squared": model.rsquared,
                        "adj_r_squared": model.rsquared_adj,
                    }
                )
                for var in available:
                    rows.append(
                        {
                            "target": target,
                            "transform": transform,
                            "spec": spec_name,
                            "variable": var,
                            "coef_bps_per_1sd": model.params.get(var, np.nan),
                            "t_stat_hac": model.tvalues.get(var, np.nan),
                            "p_value_hac": model.pvalues.get(var, np.nan),
                            "n_months": int(model.nobs),
                            "r_squared": model.rsquared,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(model_summaries)


def make_figures(monthly: pd.DataFrame) -> tuple[Path, Path]:
    ts_path = FIGURE_DIR / "eq5_ust_extension_monthly_timeseries.png"
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
    axes[0].plot(monthly["date"], monthly["x_govt_raw_bps_5y"], label="G10 median UST convenience 5Y", lw=1.2)
    axes[0].plot(monthly["date"], monthly["x_govt_raw_bps_10y"], label="G10 median UST convenience 10Y", lw=1.2)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("bps")
    axes[0].set_title("Eq.5 Direct G10 UST Convenience and Candidate Extension Controls")
    axes[0].legend(frameon=False, fontsize=8)

    for col, label in [("move_index", "MOVE"), ("bbdxy_index", "BBDXY"), ("usgg5yr_bid_ask_spread_bps", "5Y bid-ask proxy")]:
        if col in monthly.columns:
            axes[1].plot(monthly["date"], zscore(monthly[col]), label=label, lw=1.0)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("z-score")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ts_path, dpi=180)
    plt.close(fig)

    scatter_path = FIGURE_DIR / "eq5_ust_extension_control_scatter.png"
    plot_vars = [
        ("move_index", "MOVE"),
        ("bbdxy_index", "BBDXY"),
        ("us_generic_govt_5y_yield", "UST 5Y yield"),
        ("usgg5yr_bid_ask_spread_bps", "5Y bid-ask proxy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, plot_vars):
        if col not in monthly.columns:
            ax.axis("off")
            continue
        sub = monthly[["x_govt_raw_bps_5y", col]].dropna()
        ax.scatter(sub[col], sub["x_govt_raw_bps_5y"], s=12, alpha=0.55)
        ax.set_xlabel(label)
        ax.set_ylabel("UST convenience 5Y, bps")
        ax.set_title(f"Level corr: {sub['x_govt_raw_bps_5y'].corr(sub[col]):.2f}")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    return ts_path, scatter_path


def write_memo(corr: pd.DataFrame, coefs: pd.DataFrame, models: pd.DataFrame, outputs: dict[str, Path]) -> Path:
    top_corr = corr[corr["target"].eq("x_govt_raw_bps_5y")].copy()
    top_corr["abs_level_corr"] = top_corr["level_corr"].abs()
    top_corr = top_corr.sort_values("abs_level_corr", ascending=False).head(4)

    combined = models[(models["target"].eq("x_govt_raw_bps_5y")) & (models["spec"].eq("combined"))]
    combined_level = combined[combined["transform"].eq("level")].iloc[0]
    combined_change = combined[combined["transform"].eq("change")].iloc[0]

    level_coefs = coefs[
        (coefs["target"].eq("x_govt_raw_bps_5y"))
        & (coefs["transform"].eq("level"))
        & (coefs["spec"].eq("combined"))
    ].sort_values("coef_bps_per_1sd", key=lambda s: s.abs(), ascending=False)

    lines = [
        "# Eq.5 UST Convenience Extension First Pass",
        "",
        "Generated: 2026-07-06",
        "",
        "## Bottom Line",
        "",
        "Using the validated Eq.5 direct G10 5Y/10Y UST-convenience measures, I built a first-pass monthly extension dataset and tested already available volatility, dollar, rates, and generic Treasury bid-ask/liquidity proxies.",
        "",
        "This is exploratory and not causal. It is meant to identify which channels are worth taking to Samim before collecting better supply/dealer-capacity data.",
        "",
        "## What The Existing Data Can Say",
        "",
        f"- In the 5Y combined level specification, adjusted R-squared is {combined_level['adj_r_squared']:.2f} over {int(combined_level['n_months'])} monthly observations.",
        f"- In the 5Y combined monthly-change specification, adjusted R-squared is {combined_change['adj_r_squared']:.2f} over {int(combined_change['n_months'])} monthly observations.",
        "- Level relationships are much stronger than monthly-change relationships, which is consistent with slow-moving convenience and funding/rates conditions.",
        "",
        "## Strongest 5Y Level Correlations",
        "",
        "| Control | Level corr | Change corr | N months |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in top_corr.iterrows():
        lines.append(
            f"| {row['control']} | {row['level_corr']:.3f} | {row['change_corr']:.3f} | {int(row['n_months'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest Combined-Spec 5Y Coefficients",
            "",
            "Coefficients are bps of UST convenience per one-standard-deviation move in the control.",
            "",
            "| Variable | Coef | HAC t-stat |",
            "| --- | ---: | ---: |",
        ]
    )
    for _, row in level_coefs.iterrows():
        lines.append(f"| {row['variable']} | {row['coef_bps_per_1sd']:.2f} | {row['t_stat_hac']:.2f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The extension can start with Eq.5 direct UST convenience as the dependent variable.",
            "- Existing controls suggest meaningful co-movement with dollar/rates/liquidity proxies, but this is not yet a structural supply-demand test.",
            "- Better extension data are still needed for the exact Samim hypothesis: Treasury supply/debt pressure and dealer balance-sheet capacity.",
            "",
            "## Outputs",
            "",
        ]
    )
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path.relative_to(PROJECT_ROOT)}`")
    out = REPORT_DIR / "Eq5_UST_Extension_First_Pass.md"
    write_lines(out, lines)
    return out


def data_gaps() -> pd.DataFrame:
    rows = [
        {
            "needed_channel": "Treasury supply",
            "status": "missing",
            "candidate_data": "Marketable Treasury debt outstanding, Treasury outstanding by maturity bucket, debt-to-GDP",
            "why_needed": "Directly tests whether increasing supply reduces UST convenience.",
        },
        {
            "needed_channel": "Dealer capacity",
            "status": "missing",
            "candidate_data": "Primary dealer Treasury positions, repo/reverse repo, balance-sheet constraints, SLR-related proxies",
            "why_needed": "Tests Samim's intermediation-capacity channel.",
        },
        {
            "needed_channel": "Treasury liquidity",
            "status": "partial",
            "candidate_data": "Current generic bid-ask proxy is available; stronger measures include depth, price impact, off/on-the-run spreads, yield-curve fitting error",
            "why_needed": "Tests whether liquidity explains convenience changes and rebound.",
        },
        {
            "needed_channel": "Rates volatility",
            "status": "available",
            "candidate_data": "MOVE Index",
            "why_needed": "Controls for market-making risk and rates uncertainty.",
        },
        {
            "needed_channel": "Risk / dollar conditions",
            "status": "available",
            "candidate_data": "VIX, BBDXY/DXY",
            "why_needed": "Separates broad risk and dollar strength from UST-specific convenience.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = build_monthly_panel()
    monthly_path = CLEANED_DIR / "eq5_ust_extension_monthly_panel.csv"
    monthly.to_csv(monthly_path, index=False)

    corr = correlation_table(monthly)
    corr_path = TABLE_DIR / "eq5_ust_extension_correlation_table.csv"
    corr.to_csv(corr_path, index=False)

    coefs, models = regression_tables(monthly)
    coefs_path = TABLE_DIR / "eq5_ust_extension_regression_coefficients.csv"
    models_path = TABLE_DIR / "eq5_ust_extension_regression_models.csv"
    coefs.to_csv(coefs_path, index=False)
    models.to_csv(models_path, index=False)

    gaps = data_gaps()
    gaps_path = TABLE_DIR / "eq5_ust_extension_data_gaps.csv"
    gaps.to_csv(gaps_path, index=False)

    ts_path, scatter_path = make_figures(monthly)

    outputs = {
        "Monthly panel": monthly_path,
        "Correlation table": corr_path,
        "Regression coefficients": coefs_path,
        "Regression model summary": models_path,
        "Data gaps": gaps_path,
        "Timeseries figure": ts_path,
        "Scatter figure": scatter_path,
    }
    memo_path = write_memo(corr, coefs, models, outputs)
    for path in [*outputs.values(), memo_path]:
        print(path)


if __name__ == "__main__":
    main()
