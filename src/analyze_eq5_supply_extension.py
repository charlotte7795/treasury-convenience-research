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
DATA_DIR = PROJECT_ROOT / "data"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"

TARGETS = ["x_govt_raw_bps_5y", "x_govt_raw_bps_10y"]
SUPPLY_VARS = [
    "federal_debt_to_gdp_pct",
    "federal_debt_to_gdp_change_12m",
    "federal_debt_total_millions",
    "federal_debt_total_growth_12m_pct",
]
CONTROL_VARS = [
    "bbdxy_index",
    "vix_index",
    "move_index",
    "us_generic_govt_5y_yield",
    "ust_10y_2y_slope",
    "usgg5yr_bid_ask_spread_bps",
]


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std()
    if not std or np.isnan(std):
        return x * np.nan
    return (x - x.mean()) / std


def load_fred_quarterly(path: Path, fred_column: str, output_column: str, monthly_index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df[output_column] = pd.to_numeric(df[fred_column], errors="coerce")
    df["date"] = df["observation_date"].dt.to_period("M").dt.to_timestamp("M")
    df = df[["date", output_column]].dropna().sort_values("date")
    out = pd.DataFrame({"date": monthly_index})
    out = out.merge(df, on="date", how="left")
    out[output_column] = out[output_column].ffill()
    return out


def build_supply_panel() -> pd.DataFrame:
    monthly = pd.read_csv(CLEANED_DIR / "eq5_ust_extension_monthly_panel.csv", parse_dates=["date"])
    monthly = monthly.sort_values("date")
    month_index = pd.date_range(monthly["date"].min(), monthly["date"].max(), freq="ME")

    debt_gdp = load_fred_quarterly(
        DATA_DIR / "fred_GFDEGDQ188S.csv",
        "GFDEGDQ188S",
        "federal_debt_to_gdp_pct",
        month_index,
    )
    debt_total = load_fred_quarterly(
        DATA_DIR / "fred_GFDEBTN.csv",
        "GFDEBTN",
        "federal_debt_total_millions",
        month_index,
    )

    panel = monthly.merge(debt_gdp, on="date", how="left").merge(debt_total, on="date", how="left")
    panel["federal_debt_to_gdp_change_12m"] = panel["federal_debt_to_gdp_pct"].diff(12)
    panel["federal_debt_total_growth_12m_pct"] = panel["federal_debt_total_millions"].pct_change(12) * 100.0

    for target in TARGETS:
        if target in panel.columns:
            panel[f"{target}_change_12m"] = panel[target].diff(12)
    for control in CONTROL_VARS:
        if control in panel.columns:
            panel[f"{control}_change_12m"] = panel[control].diff(12)
    return panel


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for var in SUPPLY_VARS:
            sub = panel[["date", target, var]].dropna()
            if len(sub) < 36:
                continue
            rows.append(
                {
                    "target": target,
                    "supply_proxy": var,
                    "n_months": len(sub),
                    "date_min": sub["date"].min().date().isoformat(),
                    "date_max": sub["date"].max().date().isoformat(),
                    "level_corr": sub[target].corr(sub[var]),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "level_corr"], key=lambda s: s.abs() if s.name == "level_corr" else s, ascending=False)


def run_regression(y: pd.Series, x: pd.DataFrame, maxlags: int = 6):
    data = pd.concat([pd.to_numeric(y, errors="coerce").rename("y"), x.apply(zscore)], axis=1).dropna()
    if len(data) < 48:
        return None, data
    model = sm.OLS(data["y"], sm.add_constant(data.drop(columns=["y"]))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )
    return model, data


def regression_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = {
        "supply_only_level": ["federal_debt_to_gdp_pct"],
        "supply_plus_controls_level": [
            "federal_debt_to_gdp_pct",
            "bbdxy_index",
            "vix_index",
            "us_generic_govt_5y_yield",
            "ust_10y_2y_slope",
            "usgg5yr_bid_ask_spread_bps",
        ],
        "supply_only_12m_change": ["federal_debt_to_gdp_change_12m"],
        "supply_plus_controls_12m_change": [
            "federal_debt_to_gdp_change_12m",
            "bbdxy_index_change_12m",
            "vix_index_change_12m",
            "us_generic_govt_5y_yield_change_12m",
            "ust_10y_2y_slope_change_12m",
            "usgg5yr_bid_ask_spread_bps_change_12m",
        ],
    }
    coef_rows = []
    model_rows = []
    for target in TARGETS:
        for spec, variables in specs.items():
            variables = [v for v in variables if v in panel.columns]
            if spec.endswith("12m_change"):
                y = panel[f"{target}_change_12m"]
            else:
                y = panel[target]
            model, data = run_regression(y, panel[variables])
            if model is None:
                continue
            model_rows.append(
                {
                    "target": target,
                    "spec": spec,
                    "n_months": int(model.nobs),
                    "date_min": panel.loc[data.index, "date"].min().date().isoformat(),
                    "date_max": panel.loc[data.index, "date"].max().date().isoformat(),
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                }
            )
            for var in variables:
                coef_rows.append(
                    {
                        "target": target,
                        "spec": spec,
                        "variable": var,
                        "coef_bps_per_1sd": model.params.get(var, np.nan),
                        "t_stat_hac": model.tvalues.get(var, np.nan),
                        "p_value_hac": model.pvalues.get(var, np.nan),
                        "n_months": int(model.nobs),
                        "r_squared": model.rsquared,
                    }
                )
    return pd.DataFrame(coef_rows), pd.DataFrame(model_rows)


def make_figure(panel: pd.DataFrame) -> Path:
    out = FIGURE_DIR / "eq5_ust_supply_extension.png"
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)

    axes[0].plot(panel["date"], panel["x_govt_raw_bps_5y"], label="G10 median UST convenience 5Y", lw=1.2)
    axes[0].plot(panel["date"], panel["x_govt_raw_bps_10y"], label="G10 median UST convenience 10Y", lw=1.2)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("bps")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Eq.5 Direct UST Convenience and FRED Federal Debt Proxy")

    ax2 = axes[1]
    ax2.plot(panel["date"], panel["federal_debt_to_gdp_pct"], color="#b45536", label="Federal debt / GDP", lw=1.2)
    ax2.set_ylabel("% of GDP")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2b = ax2.twinx()
    ax2b.plot(
        panel["date"],
        panel["federal_debt_to_gdp_change_12m"],
        color="#5d7896",
        label="12m change",
        lw=1.0,
        alpha=0.8,
    )
    ax2b.axhline(0, color="black", lw=0.6)
    ax2b.set_ylabel("percentage points")
    ax2b.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_memo(panel: pd.DataFrame, corr: pd.DataFrame, coefs: pd.DataFrame, models: pd.DataFrame, outputs: dict[str, Path]) -> Path:
    level_model = models[
        (models["target"].eq("x_govt_raw_bps_5y")) & (models["spec"].eq("supply_plus_controls_level"))
    ].iloc[0]
    change_model = models[
        (models["target"].eq("x_govt_raw_bps_5y")) & (models["spec"].eq("supply_plus_controls_12m_change"))
    ].iloc[0]
    supply_coef = coefs[
        (coefs["target"].eq("x_govt_raw_bps_5y"))
        & (coefs["spec"].eq("supply_plus_controls_level"))
        & (coefs["variable"].eq("federal_debt_to_gdp_pct"))
    ].iloc[0]
    supply_change_coef = coefs[
        (coefs["target"].eq("x_govt_raw_bps_5y"))
        & (coefs["spec"].eq("supply_plus_controls_12m_change"))
        & (coefs["variable"].eq("federal_debt_to_gdp_change_12m"))
    ].iloc[0]
    corr_row = corr[
        (corr["target"].eq("x_govt_raw_bps_5y")) & (corr["supply_proxy"].eq("federal_debt_to_gdp_pct"))
    ].iloc[0]

    latest = panel[["date", "federal_debt_to_gdp_pct", "x_govt_raw_bps_5y"]].dropna().iloc[-1]
    lines = [
        "# Eq.5 UST Supply Extension First Pass",
        "",
        "Generated: 2026-07-07",
        "",
        "## Bottom Line",
        "",
        "I added FRED federal debt-to-GDP and total federal debt as low-frequency supply proxies to the validated Eq.5 direct G10 5Y/10Y UST-convenience panel.",
        "",
        "This is still exploratory. Federal debt-to-GDP is a broad fiscal-supply proxy, not the exact marketable Treasury supply by maturity that would be ideal for the paper extension.",
        "",
        "## Main Readout",
        "",
        f"- Latest merged observation: {latest['date'].date().isoformat()}, debt-to-GDP {latest['federal_debt_to_gdp_pct']:.1f}%, G10 5Y UST convenience {latest['x_govt_raw_bps_5y']:.1f} bps.",
        f"- Level correlation between G10 5Y UST convenience and debt-to-GDP: {corr_row['level_corr']:.3f} over {int(corr_row['n_months'])} months.",
        f"- In the 5Y supply-plus-controls level specification, adjusted R-squared is {level_model['adj_r_squared']:.2f}; debt-to-GDP coefficient is {supply_coef['coef_bps_per_1sd']:.2f} bps per 1 SD, HAC t-stat {supply_coef['t_stat_hac']:.2f}.",
        f"- In the 5Y supply-plus-controls 12-month-change specification, adjusted R-squared is {change_model['adj_r_squared']:.2f}; debt-to-GDP change coefficient is {supply_change_coef['coef_bps_per_1sd']:.2f} bps per 1 SD, HAC t-stat {supply_change_coef['t_stat_hac']:.2f}.",
        "",
        "## Interpretation",
        "",
        "- The broad supply proxy can be used as a first-pass fiscal-pressure control.",
        "- A strong level relationship would not by itself prove the supply channel, because debt-to-GDP and UST convenience are both persistent time series.",
        "- The next serious version should replace or complement this with marketable Treasury outstanding, maturity-specific outstanding, and dealer-capacity variables.",
        "",
        "## Outputs",
        "",
    ]
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path.relative_to(PROJECT_ROOT)}`")
    out = REPORT_DIR / "Eq5_UST_Supply_Extension_First_Pass.md"
    write_lines(out, lines)
    return out


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_supply_panel()
    panel_path = CLEANED_DIR / "eq5_ust_supply_extension_monthly_panel.csv"
    panel.to_csv(panel_path, index=False)

    corr = correlation_table(panel)
    corr_path = TABLE_DIR / "eq5_ust_supply_extension_correlations.csv"
    corr.to_csv(corr_path, index=False)

    coefs, models = regression_tables(panel)
    coefs_path = TABLE_DIR / "eq5_ust_supply_extension_regression_coefficients.csv"
    models_path = TABLE_DIR / "eq5_ust_supply_extension_regression_models.csv"
    coefs.to_csv(coefs_path, index=False)
    models.to_csv(models_path, index=False)

    figure_path = make_figure(panel)
    outputs = {
        "Supply monthly panel": panel_path,
        "Supply correlations": corr_path,
        "Supply regression coefficients": coefs_path,
        "Supply regression models": models_path,
        "Supply figure": figure_path,
    }
    memo_path = write_memo(panel, corr, coefs, models, outputs)

    for path in [*outputs.values(), memo_path]:
        print(path)


if __name__ == "__main__":
    main()
