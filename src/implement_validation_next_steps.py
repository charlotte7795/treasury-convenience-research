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

CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"

G10_TENORS_FOR_3M_CHECK = ["3m", "1y", "5y", "10y", "20y", "30y"]
EQ6_COMPONENTS = [
    "x_govt_raw_bps",
    "x_rf_raw_bps",
    "rho_raw_bps",
    "us_swap_spread_bps",
    "foreign_swap_spread_bps",
    "eq6_residual_bps",
]


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_csv(path: Path, date_cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=date_cols or [], low_memory=False)


def safe_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_lines(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def build_validation_register() -> tuple[Path, Path]:
    workbook = read_csv(TABLE_DIR / "downloaded_bloomberg_workbook_summary.csv")
    g10 = read_csv(TABLE_DIR / "raw_g10_5y_vs_public_comparison.csv")
    fig3 = read_csv(TABLE_DIR / "raw_figure3_long_tenors_vs_public_comparison.csv")
    em = read_csv(TABLE_DIR / "raw_em_5y_irs_bs_diagnostic_comparison.csv")
    eq6 = read_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_validation.csv")
    identity = read_csv(TABLE_DIR / "public_identity_check.csv")

    workbook_blocks = int(workbook["ticker_blocks"].sum())
    workbook_empty = int(workbook["no_numeric_blocks"].sum())
    best_identity = identity[identity["best_match"].eq(True)].iloc[0]
    eq6_govt = eq6[eq6["check_type"].eq("x_govt_vs_public_combined")]
    eq6_rho = eq6[eq6["check_type"].eq("rho_vs_public_combined")]
    eq6_resid = eq6[eq6["check_type"].eq("eq6_identity_residual")]

    rows = [
        {
            "block": "Bloomberg workbook QC",
            "status": "PASS_WITH_EM_GAPS",
            "evidence": f"{workbook_blocks - workbook_empty}/{workbook_blocks} ticker blocks non-empty; {workbook_empty} no-numeric blocks, all EM-related",
            "next_action": "Keep no-numeric EM tickers in review bucket.",
            "claim_level": "data_availability",
        },
        {
            "block": "Public processed identity",
            "status": "VALIDATED",
            "evidence": f"best-match mean absolute identity error {best_identity['mean_absolute_error']:.8f} bps",
            "next_action": "Use as public benchmark unit/sign convention.",
            "claim_level": "validated",
        },
        {
            "block": "G10 5Y government CIP",
            "status": "VALIDATED",
            "evidence": f"{g10['status'].eq('GOOD_MATCH').sum()}/{len(g10)} GOOD_MATCH; min corr {g10['corr'].min():.3f}; max MAE {g10['mae_bps'].max():.2f} bps",
            "next_action": "Use as main validated Treasury-convenience dependent variable.",
            "claim_level": "validated_raw_replication",
        },
        {
            "block": "Figure 3 long-tenor G10",
            "status": "MOSTLY_VALIDATED",
            "evidence": f"{fig3['status'].eq('GOOD_MATCH').sum()}/{len(fig3)} GOOD_MATCH; CHECK rows: {', '.join((fig3.loc[fig3['status'].ne('GOOD_MATCH'), 'currency'] + ' ' + fig3.loc[fig3['status'].ne('GOOD_MATCH'), 'tenor']).tolist())}",
            "next_action": "Keep 3M excluded; review JPY 20Y/30Y and NOK 1Y conventions.",
            "claim_level": "partial_raw_replication",
        },
        {
            "block": "EM 5Y",
            "status": "DIAGNOSTIC_ONLY",
            "evidence": "; ".join(f"{k}: {v}" for k, v in em["status"].value_counts().to_dict().items()),
            "next_action": "Split by instrument class before final EM claims.",
            "claim_level": "diagnostic",
        },
        {
            "block": "Equation 1 / 6 G10 5Y/10Y",
            "status": "FIRST_PASS_COMPONENT_VALIDATION_PENDING",
            "evidence": f"{eq6_govt['status'].eq('GOOD_MATCH').sum()}/{len(eq6_govt)} govt-CIP GOOD_MATCH; {eq6_rho['status'].eq('DIAGNOSTIC_GOOD').sum()}/{len(eq6_rho)} rho diagnostic-good; {eq6_resid['status'].eq('IDENTITY_OK').sum()}/{len(eq6_resid)} identity OK",
            "next_action": "Add component-level validation for x_rf, swap spreads, and regime breaks.",
            "claim_level": "first_pass",
        },
    ]
    out = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "validation_register.csv"
    out.to_csv(csv_path, index=False)

    lines = [
        "# Validation Register",
        "",
        "Generated: 2026-06-21",
        "",
        "| Block | Status | Evidence | Next action | Claim level |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['block']} | {row['status']} | {row['evidence']} | {row['next_action']} | {row['claim_level']} |"
        )
    md_path = REPORT_DIR / "Validation_Register.md"
    write_lines(md_path, lines)
    return csv_path, md_path


def selected_eq6_panel() -> pd.DataFrame:
    panel = read_csv(CLEANED_DIR / "raw_g10_benchmark_eq6_5y10y_panel.csv", ["date", "break_date_used"])
    return panel[panel["is_selected_regime_by_public_break"].eq(True)].copy()


def eq6_component_validation() -> tuple[Path, Path, Path, Path, Path]:
    panel = selected_eq6_panel()

    summary_rows = []
    for keys, sub in panel.groupby(["tenor", "regime"]):
        tenor, regime = keys
        row = {
            "tenor": tenor,
            "regime": regime,
            "n": int(len(sub)),
            "date_min": sub["date"].min().date().isoformat(),
            "date_max": sub["date"].max().date().isoformat(),
        }
        for col in EQ6_COMPONENTS:
            x = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[f"{col}_mean"] = x.mean()
            row[f"{col}_median"] = x.median()
            row[f"{col}_std"] = x.std()
            row[f"{col}_p25"] = x.quantile(0.25)
            row[f"{col}_p75"] = x.quantile(0.75)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = TABLE_DIR / "eq1_eq6_component_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    break_rows = []
    for (currency, tenor), sub in panel.groupby(["currency", "tenor"]):
        break_dates = sub["break_date_used"].dropna().unique()
        if len(break_dates) == 0:
            continue
        break_date = pd.Timestamp(break_dates[0])
        sub = sub.sort_values("date")
        pre = sub[sub["date"] < break_date].tail(21)
        post = sub[sub["date"] >= break_date].head(21)
        if len(pre) < 5 or len(post) < 5:
            continue
        row = {
            "currency": currency,
            "tenor": tenor,
            "break_date": break_date.date().isoformat(),
            "n_pre": len(pre),
            "n_post": len(post),
        }
        for col in EQ6_COMPONENTS:
            row[f"{col}_pre_mean"] = pre[col].mean()
            row[f"{col}_post_mean"] = post[col].mean()
            row[f"{col}_jump_post_minus_pre"] = post[col].mean() - pre[col].mean()
        break_rows.append(row)
    breaks = pd.DataFrame(break_rows)
    break_path = TABLE_DIR / "eq1_eq6_regime_break_component_jumps.csv"
    breaks.to_csv(break_path, index=False)

    daily = panel.groupby(["date", "tenor"], as_index=False)[EQ6_COMPONENTS].median()
    consistency_rows = []
    for col in EQ6_COMPONENTS:
        wide = daily.pivot(index="date", columns="tenor", values=col)
        if {"5y", "10y"}.issubset(wide.columns):
            sub = wide[["5y", "10y"]].dropna()
            consistency_rows.append(
                {
                    "component": col,
                    "n_overlap": len(sub),
                    "date_min": sub.index.min().date().isoformat(),
                    "date_max": sub.index.max().date().isoformat(),
                    "corr_5y_10y_level": sub["5y"].corr(sub["10y"]),
                    "corr_5y_10y_daily_change": sub["5y"].diff().corr(sub["10y"].diff()),
                    "mean_5y": sub["5y"].mean(),
                    "mean_10y": sub["10y"].mean(),
                }
            )
    consistency = pd.DataFrame(consistency_rows)
    consistency_path = TABLE_DIR / "eq1_eq6_5y10y_component_consistency.csv"
    consistency.to_csv(consistency_path, index=False)

    fig_path = FIGURE_DIR / "eq1_eq6_component_validation_5y10y.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    colors = {
        "x_govt_raw_bps": "#1f77b4",
        "x_rf_raw_bps": "#ff7f0e",
        "rho_raw_bps": "#9467bd",
        "us_swap_spread_bps": "#2ca02c",
        "foreign_swap_spread_bps": "#d62728",
    }
    for ax, tenor in zip(axes, ["5y", "10y"]):
        sub = daily[daily["tenor"].eq(tenor)].sort_values("date")
        for col in colors:
            ax.plot(sub["date"], sub[col].rolling(21, min_periods=5).mean(), label=col.replace("_bps", ""), lw=1.0, color=colors[col])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"G10 median Equation 1 / 6 components, {tenor.upper()}")
        ax.set_ylabel("bps")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    validation = read_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_validation.csv")
    govt_good = int(((validation["check_type"] == "x_govt_vs_public_combined") & validation["status"].eq("GOOD_MATCH")).sum())
    rho_good = int(((validation["check_type"] == "rho_vs_public_combined") & validation["status"].eq("DIAGNOSTIC_GOOD")).sum())
    resid_max = validation.loc[validation["check_type"].eq("eq6_identity_residual"), "mae_bps"].max()
    largest_jump = breaks.assign(abs_jump=breaks["x_govt_raw_bps_jump_post_minus_pre"].abs()).sort_values("abs_jump", ascending=False).head(3)
    xrf_consistency = consistency[consistency["component"].eq("x_rf_raw_bps")].iloc[0]
    lines = [
        "# Equation 1 / Equation 6 Component Validation Memo",
        "",
        "Generated: 2026-06-21",
        "",
        "## Bottom Line",
        "",
        f"- Government-CIP raw checks remain strong: {govt_good}/20 selected G10 5Y/10Y checks are `GOOD_MATCH` against public `cip_govt`.",
        f"- Raw forward-premium checks remain diagnostic-good: {rho_good}/20 selected checks are `DIAGNOSTIC_GOOD` against public `rho` converted to bps.",
        f"- Equation 6 residual max MAE is {resid_max:.2e} bps, confirming internal algebraic consistency.",
        "- This still does not make Equation 1 final, because `x_rf_raw_bps` lacks a complete direct public benchmark-CIP validation column in the current cleaned public CSV.",
        "",
        "## Component Checks Added",
        "",
        f"- Component summary table: `{rel(summary_path)}`",
        f"- Regime-break jump table: `{rel(break_path)}`",
        f"- 5Y/10Y component consistency table: `{rel(consistency_path)}`",
        f"- Component plot: `{rel(fig_path)}`",
        "",
        "## 5Y/10Y Consistency",
        "",
        f"- `x_rf_raw_bps` 5Y/10Y level correlation: {xrf_consistency['corr_5y_10y_level']:.3f}.",
        f"- `x_rf_raw_bps` 5Y/10Y daily-change correlation: {xrf_consistency['corr_5y_10y_daily_change']:.3f}.",
        "",
        "## Largest Treasury-Convenience Regime-Break Jumps",
        "",
        "| Currency | Tenor | Break date | Jump in x_govt, bps |",
        "| --- | --- | --- | ---: |",
    ]
    for _, row in largest_jump.iterrows():
        lines.append(
            f"| {row['currency']} | {row['tenor']} | {row['break_date']} | {row['x_govt_raw_bps_jump_post_minus_pre']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The raw decomposition is internally coherent and the government-CIP side matches public data.",
            "- The benchmark-CIP / dollar-convenience side should be described as first-pass until a direct benchmark-CIP public/paper comparison is available.",
            "- Regime-break jumps should be reviewed as possible convention or transition artifacts before finalizing Equation 1 claims.",
        ]
    )
    memo_path = REPORT_DIR / "Eq1_Eq6_Component_Validation_Memo.md"
    write_lines(memo_path, lines)
    return summary_path, break_path, consistency_path, fig_path, memo_path


def period_label(date: pd.Timestamp) -> str | None:
    if pd.Timestamp("2000-01-01") <= date <= pd.Timestamp("2006-12-31"):
        return "2000-2006"
    if pd.Timestamp("2010-01-01") <= date <= pd.Timestamp("2020-12-31"):
        return "2010-2020"
    if pd.Timestamp("2021-01-01") <= date <= pd.Timestamp("2025-12-31"):
        return "2021-2025"
    return None


def analyze_3m_sensitivity() -> tuple[Path, Path, Path, Path]:
    public = read_csv(CLEANED_DIR / "public_cip_cleaned.csv", ["date"])
    g10 = public[(public["group_label"].eq("G10")) & (public["tenor"].isin(G10_TENORS_FOR_3M_CHECK))].copy()
    g10 = g10.dropna(subset=["cip_govt"])
    daily = g10.groupby(["date", "tenor"], as_index=False)["cip_govt"].median()
    daily["period"] = daily["date"].map(period_label)
    period = (
        daily.dropna(subset=["period"])
        .groupby(["tenor", "period"], as_index=False)
        .agg(n_days=("cip_govt", "count"), mean_bps=("cip_govt", "mean"), median_bps=("cip_govt", "median"))
    )
    period_path = TABLE_DIR / "three_month_sensitivity_public_g10_period_summary.csv"
    period.to_csv(period_path, index=False)

    aggregate_rows = []
    for label, tenors in [
        ("including_3m", G10_TENORS_FOR_3M_CHECK),
        ("excluding_3m", [t for t in G10_TENORS_FOR_3M_CHECK if t != "3m"]),
        ("5y_10y_only", ["5y", "10y"]),
    ]:
        sub = daily[daily["tenor"].isin(tenors)]
        agg_daily = sub.groupby("date", as_index=False)["cip_govt"].median()
        agg_daily["period"] = agg_daily["date"].map(period_label)
        agg = (
            agg_daily.dropna(subset=["period"])
            .groupby("period", as_index=False)
            .agg(n_days=("cip_govt", "count"), mean_bps=("cip_govt", "mean"), median_bps=("cip_govt", "median"))
        )
        early = agg.loc[agg["period"].eq("2000-2006"), "mean_bps"]
        recent = agg.loc[agg["period"].eq("2021-2025"), "mean_bps"]
        aggregate_rows.append(
            {
                "spec": label,
                "tenors": ",".join(tenors),
                "early_mean_bps": float(early.iloc[0]) if len(early) else np.nan,
                "recent_mean_bps": float(recent.iloc[0]) if len(recent) else np.nan,
                "recent_minus_early_bps": float(recent.iloc[0] - early.iloc[0]) if len(early) and len(recent) else np.nan,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate_path = TABLE_DIR / "three_month_sensitivity_core_conclusion.csv"
    aggregate.to_csv(aggregate_path, index=False)

    fig_path = FIGURE_DIR / "three_month_sensitivity_public_g10.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tenor in ["3m", "1y", "5y", "10y", "20y", "30y"]:
        sub = daily[daily["tenor"].eq(tenor)].sort_values("date")
        ax.plot(sub["date"], sub["cip_govt"].rolling(63, min_periods=10).mean(), lw=0.9, label=tenor)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Public G10 Treasury Convenience by Tenor: 3M Sensitivity")
    ax.set_ylabel("G10 median government CIP deviation, bps")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    ex3m = aggregate[aggregate["spec"].eq("excluding_3m")].iloc[0]
    f5y10y = aggregate[aggregate["spec"].eq("5y_10y_only")].iloc[0]
    lines = [
        "# 3M Sensitivity Note",
        "",
        "Generated: 2026-06-21",
        "",
        "## Bottom Line",
        "",
        "- The public processed government-CIP data show that the Treasury-convenience decline does not rely only on the 3M tenor.",
        f"- Excluding 3M, the public G10 cross-tenor mean moves by {ex3m['recent_minus_early_bps']:.2f} bps from 2000-2006 to 2021-2025.",
        f"- Using only 5Y/10Y, the same early-to-recent change is {f5y10y['recent_minus_early_bps']:.2f} bps.",
        "- Therefore, unresolved raw 3M construction should not block work focused on the decline in Treasury convenience.",
        "",
        "## Important Caveat",
        "",
        "- This conclusion is about government-CIP / Treasury-convenience measures.",
        "- 3M remains important for fully validating benchmark CIP / dollar convenience, because short-maturity Equation 1 construction uses spot and FX forwards rather than long-tenor swap/basis construction.",
        "",
        "## Outputs",
        "",
        f"- Period summary: `{rel(period_path)}`",
        f"- Core-conclusion comparison: `{rel(aggregate_path)}`",
        f"- Figure: `{rel(fig_path)}`",
    ]
    memo_path = REPORT_DIR / "Three_Month_Sensitivity_Note.md"
    write_lines(memo_path, lines)
    return period_path, aggregate_path, fig_path, memo_path


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std()
    if not std or pd.isna(std):
        return x * np.nan
    return (x - x.mean()) / std


def controls_wide() -> pd.DataFrame:
    values = read_csv(CLEANED_DIR / "downloaded_bloomberg_values_long.csv", ["date"])
    values = values[values["workbook"].isin(["macro_controls", "treasury_liquidity_dealer"])].copy()
    wide = values.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    wide.columns.name = None
    for tenor in ["2yr", "5yr", "10yr", "30yr"]:
        bid = f"usgg{tenor}_px_bid"
        ask = f"usgg{tenor}_px_ask"
        if bid in wide.columns and ask in wide.columns:
            wide[f"usgg{tenor}_bid_ask_spread_bps"] = (wide[ask] - wide[bid]).abs() * 100
    return wide.reset_index()


def analyze_recent_rebound() -> tuple[Path, Path, Path, Path]:
    panel = selected_eq6_panel()
    panel = panel[panel["tenor"].eq("5y")].copy()
    panel["diff_y_raw_bps"] = 100 * (panel["foreign_govt_yield"] - panel["usd_govt_yield"])
    cols = [
        "x_govt_raw_bps",
        "diff_y_raw_bps",
        "rho_raw_bps",
        "x_rf_raw_bps",
        "us_swap_spread_bps",
        "foreign_swap_spread_bps",
    ]
    daily = panel.groupby("date", as_index=False)[cols].median().sort_values("date")
    recent = daily[daily["date"].ge(pd.Timestamp("2023-01-01"))].copy()
    for col in cols:
        recent[f"{col}_roll21"] = recent[col].rolling(21, min_periods=5).mean()

    valid = recent.dropna(subset=["x_govt_raw_bps_roll21"])
    trough_row = valid.loc[valid["x_govt_raw_bps_roll21"].idxmin()]
    latest_row = valid.iloc[-1]
    driver_rows = []
    for col in cols:
        driver_rows.append(
            {
                "component": col,
                "trough_date": trough_row["date"].date().isoformat(),
                "latest_date": latest_row["date"].date().isoformat(),
                "trough_value_bps": trough_row[f"{col}_roll21"],
                "latest_value_bps": latest_row[f"{col}_roll21"],
                "latest_minus_trough_bps": latest_row[f"{col}_roll21"] - trough_row[f"{col}_roll21"],
            }
        )
    drivers = pd.DataFrame(driver_rows)
    drivers_path = TABLE_DIR / "recent_rebound_5y_driver_decomposition.csv"
    drivers.to_csv(drivers_path, index=False)

    controls = controls_wide()
    merged = recent.merge(controls, on="date", how="left")
    control_vars = [
        "move_index",
        "vix_index",
        "bbdxy_index",
        "dxy_currency_index",
        "us_generic_govt_5y_yield",
        "us_generic_govt_10y_yield",
        "usgg5yr_bid_ask_spread_bps",
        "usgg10yr_bid_ask_spread_bps",
    ]
    corr_rows = []
    for var in control_vars:
        if var not in merged.columns:
            continue
        sub = merged[["x_govt_raw_bps", var]].dropna()
        if len(sub) < 100:
            continue
        corr_rows.append(
            {
                "control_or_proxy": var,
                "n": len(sub),
                "corr_level": sub["x_govt_raw_bps"].corr(sub[var]),
                "corr_daily_change": sub["x_govt_raw_bps"].diff().corr(sub[var].diff()),
            }
        )
    corr = pd.DataFrame(corr_rows).sort_values("corr_level", key=lambda s: s.abs(), ascending=False)
    corr_path = TABLE_DIR / "recent_rebound_5y_control_correlations.csv"
    corr.to_csv(corr_path, index=False)

    fig_path = FIGURE_DIR / "recent_rebound_5y_treasury_convenience_diagnostic.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(recent["date"], recent["x_govt_raw_bps_roll21"], label="x_govt / Treasury convenience", lw=1.5)
    axes[0].plot(recent["date"], recent["diff_y_raw_bps_roll21"], label="foreign govt yield - UST yield", lw=1.0)
    axes[0].plot(recent["date"], recent["rho_raw_bps_roll21"], label="rho / forward premium", lw=1.0)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].axvline(trough_row["date"], color="gray", lw=0.8, ls="--")
    axes[0].set_title("G10 Median 5Y Treasury Convenience Rebound Diagnostic")
    axes[0].set_ylabel("bps, 21-day rolling mean")
    axes[0].legend(frameon=False, fontsize=8)

    for var, label in [("move_index", "MOVE"), ("vix_index", "VIX"), ("bbdxy_index", "BBDXY")]:
        if var in merged.columns:
            axes[1].plot(merged["date"], zscore(merged[var]).rolling(21, min_periods=5).mean(), label=label, lw=1.0)
    axes[1].plot(merged["date"], zscore(merged["x_govt_raw_bps"]).rolling(21, min_periods=5).mean(), label="x_govt z-score", lw=1.2)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("z-score")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    dx = drivers.set_index("component")["latest_minus_trough_bps"]
    lines = [
        "# Recent Treasury-Convenience Rebound Diagnostic",
        "",
        "Generated: 2026-06-21",
        "",
        "## Bottom Line",
        "",
        f"- Latest available raw G10 5Y selected-regime date: {latest_row['date'].date().isoformat()}.",
        f"- 2023-onward trough in 21-day rolling G10 median 5Y Treasury convenience: {trough_row['date'].date().isoformat()}.",
        f"- From trough to latest, `x_govt_raw_bps` moved by {dx['x_govt_raw_bps']:.2f} bps.",
        f"- The corresponding diagnostic component changes are: `diff_y_raw_bps` {dx['diff_y_raw_bps']:.2f} bps and `rho_raw_bps` {dx['rho_raw_bps']:.2f} bps.",
        "- Since `x_govt = diff_y - rho`, this is a mechanical decomposition, not a causal explanation.",
        "",
        "## Interpretation To Discuss",
        "",
        "- A rebound can come from foreign-minus-U.S. government yield differentials rising, forward-premium/rho falling, or both.",
        "- Current controls can describe co-movement with MOVE, VIX, dollar indexes, and generic Treasury bid-ask proxies, but they do not yet establish causality.",
        "- This is the right bridge from replication into the later liquidity / dealer-capacity extension.",
        "",
        "## Outputs",
        "",
        f"- Driver decomposition: `{rel(drivers_path)}`",
        f"- Control correlations: `{rel(corr_path)}`",
        f"- Figure: `{rel(fig_path)}`",
    ]
    memo_path = REPORT_DIR / "Recent_Treasury_Convenience_Rebound_Diagnostic.md"
    write_lines(memo_path, lines)
    return drivers_path, corr_path, fig_path, memo_path


def build_master_memo(outputs: dict[str, Path]) -> Path:
    lines = [
        "# Validation-First Implementation Memo",
        "",
        "Generated: 2026-06-21",
        "",
        "## What Was Implemented",
        "",
        "1. Re-ran the current QC and replication baseline.",
        "2. Built a validation register separating validated, partial, first-pass, and diagnostic claims.",
        "3. Added Equation 1 / Equation 6 component validation outputs.",
        "4. Added a public-data 3M sensitivity check for the Treasury-convenience decline result.",
        "5. Added a 2023-to-latest rebound diagnostic for G10 median 5Y Treasury convenience.",
        "",
        "## Main Takeaways",
        "",
        "- G10 5Y government-CIP replication remains the strongest validated block.",
        "- Equation 1 / 6 is internally coherent and matches public government-CIP/rho diagnostics, but the benchmark-CIP side remains first-pass.",
        "- The public Treasury-convenience decline result does not rely only on 3M; non-3M and 5Y/10Y versions also show large early-to-recent declines.",
        "- The recent rebound can now be discussed mechanically through `diff_y` and `rho`, but not yet causally.",
        "- EM should remain diagnostic until country-specific instrument rules are resolved.",
        "",
        "## New Outputs",
        "",
    ]
    for label, path in outputs.items():
        lines.append(f"- {label}: `{rel(path)}`")
    return write_lines(REPORT_DIR / "Validation_First_Implementation_Memo.md", lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    validation_csv, validation_md = build_validation_register()
    eq6_summary, eq6_breaks, eq6_consistency, eq6_fig, eq6_memo = eq6_component_validation()
    m3_period, m3_core, m3_fig, m3_memo = analyze_3m_sensitivity()
    rebound_drivers, rebound_corr, rebound_fig, rebound_memo = analyze_recent_rebound()

    outputs = {
        "Validation register CSV": validation_csv,
        "Validation register memo": validation_md,
        "Equation 1/6 component summary": eq6_summary,
        "Equation 1/6 regime-break checks": eq6_breaks,
        "Equation 1/6 5Y/10Y consistency": eq6_consistency,
        "Equation 1/6 component figure": eq6_fig,
        "Equation 1/6 component memo": eq6_memo,
        "3M period summary": m3_period,
        "3M core conclusion check": m3_core,
        "3M sensitivity figure": m3_fig,
        "3M sensitivity note": m3_memo,
        "Recent rebound drivers": rebound_drivers,
        "Recent rebound control correlations": rebound_corr,
        "Recent rebound figure": rebound_fig,
        "Recent rebound memo": rebound_memo,
    }
    master = build_master_memo(outputs)
    outputs["Master implementation memo"] = master
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
