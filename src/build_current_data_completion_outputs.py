from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / "output" / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std()
    if not std or np.isnan(std):
        return x * np.nan
    return (x - x.mean()) / std


def selected_eq6_daily() -> pd.DataFrame:
    panel = pd.read_csv(CLEANED_DIR / "raw_g10_benchmark_eq6_5y10y_panel.csv", parse_dates=["date"], low_memory=False)
    panel = panel[panel["is_selected_regime_by_public_break"].eq(True)].copy()
    metrics = [
        "x_govt_raw_bps",
        "x_rf_raw_bps",
        "us_swap_spread_bps",
        "foreign_swap_spread_bps",
        "eq6_rhs_bps",
    ]
    daily = panel.groupby(["date", "tenor"], as_index=False)[metrics].median()
    wide = daily.pivot(index="date", columns="tenor", values=metrics)
    wide.columns = [f"{metric}_{tenor}" for metric, tenor in wide.columns]
    return wide.reset_index().sort_values("date")


def macro_liquidity_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv(
        CLEANED_DIR / "downloaded_bloomberg_values_long.csv",
        usecols=["workbook", "date", "variable", "value"],
        parse_dates=["date"],
    )
    values = values[values["workbook"].isin(["macro_controls", "treasury_liquidity_dealer"])].copy()
    coverage = (
        values.groupby(["workbook", "variable"])
        .agg(n=("value", "count"), first_date=("date", "min"), last_date=("date", "max"))
        .reset_index()
    )
    wide = values.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    wide.columns.name = None
    for tenor in ["2yr", "5yr", "10yr", "30yr"]:
        bid = f"usgg{tenor}_px_bid"
        ask = f"usgg{tenor}_px_ask"
        if bid in wide and ask in wide:
            wide[f"usgg{tenor}_bid_ask_spread_bps"] = (wide[ask] - wide[bid]).abs() * 100.0
    return wide.reset_index().sort_values("date"), coverage


def correlation_table(eq6_daily: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    merged = eq6_daily.merge(controls, on="date", how="inner").sort_values("date")
    targets = [
        "x_govt_raw_bps_5y",
        "x_rf_raw_bps_5y",
        "x_govt_raw_bps_10y",
        "x_rf_raw_bps_10y",
    ]
    variables = [
        "vix_index",
        "move_index",
        "dxy_currency_index",
        "bbdxy_index",
        "us_generic_govt_2y_yield",
        "us_generic_govt_5y_yield",
        "us_generic_govt_10y_yield",
        "us_generic_govt_30y_yield",
        "usgg2yr_bid_ask_spread_bps",
        "usgg5yr_bid_ask_spread_bps",
        "usgg10yr_bid_ask_spread_bps",
        "usgg30yr_bid_ask_spread_bps",
    ]
    rows = []
    for target in targets:
        if target not in merged:
            continue
        for var in variables:
            if var not in merged:
                continue
            sub = merged[["date", target, var]].dropna()
            if len(sub) < 250:
                continue
            change = sub[[target, var]].diff().dropna()
            rows.append(
                {
                    "target": target,
                    "control_or_proxy": var,
                    "n_level": len(sub),
                    "date_min": sub["date"].min().date().isoformat(),
                    "date_max": sub["date"].max().date().isoformat(),
                    "corr_level": sub[target].corr(sub[var]),
                    "n_daily_change": len(change),
                    "corr_daily_change": change[target].corr(change[var]),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "control_or_proxy"])


def status_matrix() -> pd.DataFrame:
    g10 = pd.read_csv(TABLE_DIR / "raw_g10_5y_vs_public_comparison.csv")
    fig3 = pd.read_csv(TABLE_DIR / "raw_figure3_long_tenors_vs_public_comparison.csv")
    eq6 = pd.read_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_validation.csv")
    em = pd.read_csv(TABLE_DIR / "raw_em_5y_irs_bs_diagnostic_comparison.csv")
    identity = pd.read_csv(TABLE_DIR / "public_identity_check.csv")
    best_identity = identity[identity["best_match"].eq(True)].iloc[0]
    rows = [
        {
            "workstream": "Public processed identity",
            "current_status": "completed",
            "evidence": f"best-match mean absolute identity error {best_identity['mean_absolute_error']:.8f} bps",
            "can_do_with_existing_data": "Write validation section and use public data as benchmark.",
            "remaining_blocker": "",
        },
        {
            "workstream": "G10 5Y raw government CIP",
            "current_status": "validated",
            "evidence": f"{g10['status'].eq('GOOD_MATCH').sum()}/{len(g10)} GOOD_MATCH; min corr {g10['corr'].min():.3f}; max MAE {g10['mae_bps'].max():.2f} bps",
            "can_do_with_existing_data": "Use as first stable raw dependent variable; include final plot/table.",
            "remaining_blocker": "",
        },
        {
            "workstream": "Figure 3 long-tenor G10",
            "current_status": "mostly completed",
            "evidence": f"{fig3['status'].eq('GOOD_MATCH').sum()}/{len(fig3)} GOOD_MATCH",
            "can_do_with_existing_data": "Use long-tenor partial replication with convention notes.",
            "remaining_blocker": "3M FX convention; JPY 20Y/30Y and NOK 1Y convention checks.",
        },
        {
            "workstream": "Equation 1 / Equation 6 G10 5Y/10Y",
            "current_status": "first pass completed",
            "evidence": f"{((eq6.check_type == 'x_govt_vs_public_combined') & (eq6.status == 'GOOD_MATCH')).sum()}/20 govt-CIP checks; {((eq6.check_type == 'eq6_identity_residual') & (eq6.status == 'IDENTITY_OK')).sum()}/40 identity checks",
            "can_do_with_existing_data": "Write decomposition memo; use x_rf_raw_bps as first dollar-convenience proxy.",
            "remaining_blocker": "Direct public/paper benchmark-CIP validation.",
        },
        {
            "workstream": "EM 5Y diagnostic",
            "current_status": "diagnostic only",
            "evidence": ", ".join(f"{k}: {v}" for k, v in em["status"].value_counts().to_dict().items()),
            "can_do_with_existing_data": "Show constructability and validated/usable/review country split.",
            "remaining_blocker": "NDS/CCS/country-specific rules and 4 no-numeric ticker blocks.",
        },
        {
            "workstream": "Macro/liquidity exploratory",
            "current_status": "exploratory appendix",
            "evidence": "VIX, MOVE, DXY/BBDXY, generic UST yields, and generic UST bid/ask candidate fields are downloaded.",
            "can_do_with_existing_data": "Compute simple level/change correlations and z-score visuals.",
            "remaining_blocker": "Paper regression specification and debt/dealer-capacity datasets.",
        },
    ]
    return pd.DataFrame(rows)


def plot_zscore(eq6_daily: pd.DataFrame, controls: pd.DataFrame) -> Path:
    merged = eq6_daily.merge(controls, on="date", how="inner").sort_values("date")
    series = {
        "G10 median 5Y Treasury convenience": "x_govt_raw_bps_5y",
        "G10 median 5Y dollar-convenience proxy": "x_rf_raw_bps_5y",
        "MOVE": "move_index",
        "VIX": "vix_index",
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, col in series.items():
        if col not in merged:
            continue
        s = zscore(merged[col]).rolling(21, min_periods=5).mean()
        ax.plot(merged["date"], s, lw=1.0, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Existing Data Extension Preview: Convenience Measures vs Risk Proxies")
    ax.set_ylabel("Z-score, 21-day rolling mean")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIGURE_DIR / "current_data_macro_liquidity_zscore_preview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_eq6_components(eq6_daily: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, tenor in zip(axes, ["5y", "10y"]):
        for col, label in [
            (f"x_govt_raw_bps_{tenor}", "Treasury convenience / x_govt"),
            (f"x_rf_raw_bps_{tenor}", "Dollar convenience proxy / x_rf"),
            (f"us_swap_spread_bps_{tenor}", "US swap spread"),
            (f"foreign_swap_spread_bps_{tenor}", "Foreign swap spread"),
        ]:
            if col in eq6_daily:
                ax.plot(eq6_daily["date"], eq6_daily[col].rolling(21, min_periods=5).mean(), lw=0.9, label=label)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"G10 median Eq.6 components, {tenor.upper()}")
        ax.set_ylabel("bps")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIGURE_DIR / "current_data_eq6_components_5y10y.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_memo(status: pd.DataFrame, corr: pd.DataFrame, zscore_path: Path, eq6_path: Path) -> Path:
    eq6 = pd.read_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_validation.csv")
    govt_good = int(((eq6["check_type"] == "x_govt_vs_public_combined") & (eq6["status"] == "GOOD_MATCH")).sum())
    rho_good = int(((eq6["check_type"] == "rho_vs_public_combined") & (eq6["status"] == "DIAGNOSTIC_GOOD")).sum())
    identity_ok = int(((eq6["check_type"] == "eq6_identity_residual") & (eq6["status"] == "IDENTITY_OK")).sum())
    top_corr = corr.dropna(subset=["corr_level"]).copy()
    top_corr["abs_corr_level"] = top_corr["corr_level"].abs()
    top_corr = top_corr.sort_values("abs_corr_level", ascending=False).head(8)
    lines = [
        "# Existing Data Completion Memo",
        "",
        "## Bottom Line",
        "",
        "I agree with the attached analysis with one update: Equation 1 / Equation 6 first pass is already implemented for G10 5Y/10Y in the current workspace. The remaining existing-data work is therefore to package the validated results, add a cautious macro/liquidity exploratory appendix, and map the exact remaining data needs.",
        "",
        "## What Existing Data Can Support Now",
        "",
    ]
    for _, row in status.iterrows():
        lines.append(f"- **{row['workstream']}**: {row['current_status']}. {row['evidence']}")
    lines.extend(
        [
            "",
            "## Equation 1 / Equation 6 First Pass",
            "",
            f"- Selected-regime government-CIP checks: {govt_good}/20 `GOOD_MATCH`.",
            f"- Selected-regime rho diagnostics: {rho_good}/20 `DIAGNOSTIC_GOOD`.",
            f"- Equation 6 identity residual checks: {identity_ok}/40 `IDENTITY_OK`.",
            "- Interpretation: `x_rf_raw_bps` is a first raw dollar-convenience proxy validated through component logic and Eq.6 identity. It is not yet a direct public benchmark-CIP replication.",
            "",
            "## Macro / Liquidity Exploratory Preview",
            "",
            "This is not a paper replication yet. It is a low-stakes extension preview using already downloaded VIX, MOVE, dollar index, UST yields, and generic UST bid/ask candidate fields.",
            "",
            "| Target | Control/proxy | Level corr | Daily-change corr | N |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in top_corr.iterrows():
        lines.append(
            f"| {row['target']} | {row['control_or_proxy']} | {row['corr_level']:.3f} | {row['corr_daily_change']:.3f} | {int(row['n_level'])} |"
        )
    lines.extend(
        [
            "",
            "## Generated Outputs",
            "",
            f"- `{status_path_rel(TABLE_DIR / 'current_data_completion_status.csv')}`",
            f"- `{status_path_rel(TABLE_DIR / 'macro_liquidity_exploratory_correlations.csv')}`",
            f"- `{status_path_rel(TABLE_DIR / 'macro_liquidity_variable_coverage.csv')}`",
            f"- `{status_path_rel(zscore_path)}`",
            f"- `{status_path_rel(eq6_path)}`",
            "",
            "## Remaining Data Categories",
            "",
            "1. 3M FX spot / forward / benchmark-rate convention data for Figure 1 and Figure 3 3M.",
            "2. EM NDS / CCS / country-specific construction data for full raw Figure 2 and Table 2 EM.",
            "3. Debt supply, debt-to-GDP, official holdings, and tenor-specific Treasury outstanding for Figure 9 and Tables 6/7.",
            "4. Treasury liquidity and dealer-capacity proxies for the Samim extension.",
            "5. Event dates / event-window definitions for Figure 7.",
        ]
    )
    out = REPORT_DIR / "Existing_Data_Completion_Memo.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def status_path_rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    eq6_daily = selected_eq6_daily()
    controls, coverage = macro_liquidity_daily()
    corr = correlation_table(eq6_daily, controls)
    status = status_matrix()
    zfig = plot_zscore(eq6_daily, controls)
    efig = plot_eq6_components(eq6_daily)
    status.to_csv(TABLE_DIR / "current_data_completion_status.csv", index=False)
    coverage.to_csv(TABLE_DIR / "macro_liquidity_variable_coverage.csv", index=False)
    corr.to_csv(TABLE_DIR / "macro_liquidity_exploratory_correlations.csv", index=False)
    memo = write_memo(status, corr, zfig, efig)
    print(TABLE_DIR / "current_data_completion_status.csv")
    print(TABLE_DIR / "macro_liquidity_variable_coverage.csv")
    print(TABLE_DIR / "macro_liquidity_exploratory_correlations.csv")
    print(zfig)
    print(efig)
    print(memo)


if __name__ == "__main__":
    main()
