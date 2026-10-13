"""Create first-pass paper replication assets.

The focus is to separate three layers clearly:
1. What can be reproduced from public processed CIP data.
2. What can be checked with the current raw Bloomberg pilot.
3. What still requires additional raw data or macro data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_DIR = OUTPUT_DIR / "report"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
CLEANED_DIR = OUTPUT_DIR / "cleaned"

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402


TENOR_ORDER = ["3m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]
TENOR_YEARS = {
    "3m": 0.25,
    "1y": 1.0,
    "2y": 2.0,
    "3y": 3.0,
    "5y": 5.0,
    "7y": 7.0,
    "10y": 10.0,
    "20y": 20.0,
    "30y": 30.0,
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR, FIGURE_DIR, CLEANED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_public() -> pd.DataFrame:
    candidates = [
        CLEANED_DIR / "public_cip_cleaned.csv",
        OUTPUT_DIR / "cleaned_cip_dataset.csv",
        PROJECT_ROOT / "data" / "cip_dataset_v4.csv",
        PROJECT_ROOT / "CIP data" / "cip_dataset_v4.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Could not find public CIP dataset.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().mean() > 0.5:
        df["date"] = pd.to_datetime(df["date"], format="%d%b%Y", errors="coerce")
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df["group"] = df["group"].astype(str).str.lower().str.strip()
    df["group_label"] = df["group"].map({"g10": "G10", "eme": "EM"}).fillna(df["group"].str.upper())
    df["tenor"] = df["tenor"].astype(str).str.lower().str.strip()
    for col in [
        "diff_y",
        "rho",
        "cip_govt",
        "rho_ibor",
        "rho_sofr",
        "cip_govt_ibor",
        "cip_govt_sofr",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"])


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def make_replication_matrix() -> pd.DataFrame:
    rows = [
        {
            "paper_item": "Figure 1",
            "paper_title": "Benchmark CIP Deviations: 3-Month IBOR-based vs Alternative Rate-Based",
            "paper_target": "Benchmark / risk-free CIP deviations under IBOR and alternative reference rates.",
            "public_processed_status": "Partial only",
            "current_raw_status": "Not yet available",
            "current_output": "Related IBOR/SOFR government-CIP comparison only; not exact Figure 1.",
            "data_needed_for_full_raw_replication": "3M spot FX, 3M forward/FX swap points, benchmark rates for G10 currencies, USD benchmark rates, convention metadata.",
            "priority": "Medium",
            "notes": "Public government-bond CIP dataset is not enough to fully reconstruct benchmark CIP.",
        },
        {
            "paper_item": "Figure 2",
            "paper_title": "The Decline of the U.S. Treasury Convenience",
            "paper_target": "Time series of government bond CIP / UST convenience.",
            "public_processed_status": "Ready",
            "current_raw_status": "Partial: EUR/GBP 5Y/10Y raw pilot",
            "current_output": "replication_figure2_public_g10_median_5y_10y.png; raw_public_latest_extension_5y_10y.png",
            "data_needed_for_full_raw_replication": "Full raw G10 government yields, IRS/OIS rates, and cross-currency basis swaps across tenors.",
            "priority": "High",
            "notes": "Public processed data provide the benchmark; raw pilot validates EUR/GBP subset and extends selected series to 2026.",
        },
        {
            "paper_item": "Figure 3",
            "paper_title": "Term Structure of the U.S. Treasury Premium of G10 Currency Countries",
            "paper_target": "Term structure of G10 government bond CIP / UST premium across maturities.",
            "public_processed_status": "Ready",
            "current_raw_status": "Partial: EUR/GBP 5Y/10Y raw pilot only",
            "current_output": "replication_figure3_public_term_structure_g10.png",
            "data_needed_for_full_raw_replication": "Raw G10 data for all available tenors: 3M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y.",
            "priority": "High",
            "notes": "Good first target because the public data include multiple tenors.",
        },
        {
            "paper_item": "Figure 4",
            "paper_title": "The Decoupling of USD and UST Premia",
            "paper_target": "USD convenience vs UST convenience and their decomposition.",
            "public_processed_status": "Partial / insufficient",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "Benchmark CIP x_Rf, government CIP x_Govt, USD and foreign swap spreads, benchmark transition conventions.",
            "priority": "High",
            "notes": "This is central to Equation 6 and should be the next conceptual/data target after Figure 2/3/Table 2.",
        },
        {
            "paper_item": "Figure 5",
            "paper_title": "Time-Varying Loading on VIX",
            "paper_target": "Rolling / time-varying sensitivity of UST convenience to VIX.",
            "public_processed_status": "Needs external data",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "VIX daily/weekly data, regression window definition, UST convenience component series.",
            "priority": "Medium",
            "notes": "Could start after Figure 2 series is stable; VIX is obtainable but not currently in workspace.",
        },
        {
            "paper_item": "Figure 6",
            "paper_title": "Time-Varying Loading on VIX: Bilateral",
            "paper_target": "Bilateral VIX sensitivity by currency/country.",
            "public_processed_status": "Needs external data",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "VIX data, bilateral CIP series, regression specifications.",
            "priority": "Medium",
            "notes": "Requires details from the paper's regression setup.",
        },
        {
            "paper_item": "Figure 7",
            "paper_title": "Changes in U.S. Treasury Convenience following reciprocal tariffs announcement",
            "paper_target": "Event-study change in UST convenience around tariff announcement.",
            "public_processed_status": "Potentially partial",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "Event date specification, daily CIP series, ideally updated raw series beyond public-data horizon.",
            "priority": "Low",
            "notes": "Confirm relevance and event window with Samim before prioritizing.",
        },
        {
            "paper_item": "Figure 8",
            "paper_title": "Benchmark and Government CIP Deviations During Financial Distress",
            "paper_target": "Benchmark and government CIP behavior in stress windows.",
            "public_processed_status": "Government-CIP partial ready",
            "current_raw_status": "Partial: EUR/GBP raw pilot for 2022-2026 only",
            "current_output": "replication_figure8_public_distress_windows.png",
            "data_needed_for_full_raw_replication": "Benchmark CIP series plus government CIP for all stress windows.",
            "priority": "Medium",
            "notes": "Public data support a government-CIP-only stress-window figure for GFC/COVID.",
        },
        {
            "paper_item": "Figure 9",
            "paper_title": "Government debt and swap spreads",
            "paper_target": "Debt supply and swap spread relationship.",
            "public_processed_status": "Needs external data",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "Government debt / supply measures, swap spreads, macro controls.",
            "priority": "High for extension",
            "notes": "Directly related to Samim's supply and intermediation-capacity extension idea.",
        },
        {
            "paper_item": "Table 1",
            "paper_title": "IBOR and Alternative Benchmark Rates",
            "paper_target": "Reference-rate mapping by currency.",
            "public_processed_status": "Descriptive / docs-supported",
            "current_raw_status": "Partial: EUR ESTR, GBP SONIA, USD SOFR, JPY OIS available in pilot files",
            "current_output": "No table yet; can be written from ticker supplement and raw file inventory.",
            "data_needed_for_full_raw_replication": "Full G10 benchmark-rate mapping and ticker conventions.",
            "priority": "Medium",
            "notes": "Useful as a report appendix.",
        },
        {
            "paper_item": "Table 2",
            "paper_title": "Summary statistics of the 5-year Government Bond CIP Deviations",
            "paper_target": "Summary statistics for 5Y government-bond CIP deviations.",
            "public_processed_status": "Ready",
            "current_raw_status": "Partial: EUR/GBP 5Y raw pilot",
            "current_output": "replication_table2_public_5y_summary.csv; replication_table2_public_5y_major_currencies.csv",
            "data_needed_for_full_raw_replication": "Full G10 raw 5Y reconstruction for country-level raw replication.",
            "priority": "High",
            "notes": "This is the easiest first table to put in the PDF/Word report.",
        },
        {
            "paper_item": "Table 3",
            "paper_title": "Summary statistics of the 5-year Benchmark CIP Deviations",
            "paper_target": "Summary statistics for 5Y benchmark / risk-free CIP.",
            "public_processed_status": "Insufficient",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "Risk-free benchmark rates, FX hedge premia, benchmark CIP construction.",
            "priority": "High after raw benchmark data",
            "notes": "Needed for dollar-vs-Treasury privilege decomposition.",
        },
        {
            "paper_item": "Table 4",
            "paper_title": "Decomposition of Government Bond CIP Deviations",
            "paper_target": "Equation 6 decomposition: UST convenience into USD convenience and swap-spread components.",
            "public_processed_status": "Insufficient",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "x_Rf, U.S. swap spreads, foreign swap spreads, government yields.",
            "priority": "High",
            "notes": "This is likely the most important next raw-data target.",
        },
        {
            "paper_item": "Table 5",
            "paper_title": "Risk Sensitivity of U.S. Treasury Convenience and its Components",
            "paper_target": "Regression of convenience components on risk variables.",
            "public_processed_status": "Needs external data",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "VIX/risk measures, component series from Table 4, regression controls.",
            "priority": "Medium",
            "notes": "Dependent on Table 4 decomposition.",
        },
        {
            "paper_item": "Table 6",
            "paper_title": "Effect of government bond supply on the U.S. Treasury Convenience",
            "paper_target": "Quarterly panel regression on government bond supply.",
            "public_processed_status": "Needs external data",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "Government bond supply, debt-to-GDP or debt outstanding, quarterly aggregation, controls.",
            "priority": "High for extension",
            "notes": "Closest to Samim's proposed supply / dealer-capacity extension.",
        },
        {
            "paper_item": "Table 7",
            "paper_title": "Effect of tenor-specific U.S. Treasury supply on the U.S. Treasury Convenience",
            "paper_target": "Regression using tenor-specific Treasury supply.",
            "public_processed_status": "Needs external data",
            "current_raw_status": "Not yet available",
            "current_output": "No exact replication yet.",
            "data_needed_for_full_raw_replication": "Tenor-specific Treasury supply / issuance / outstanding data and panel controls.",
            "priority": "High for extension",
            "notes": "Important for future Treasury supply and market-structure contribution.",
        },
    ]
    return pd.DataFrame(rows)


def period_label(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2007-08-01"):
        return "Pre-GFC"
    if date <= pd.Timestamp("2009-06-30"):
        return "GFC"
    if date <= pd.Timestamp("2020-02-28"):
        return "Post-GFC / pre-COVID"
    if date <= pd.Timestamp("2020-06-30"):
        return "COVID shock"
    if date <= pd.Timestamp("2021-12-31"):
        return "Post-COVID / pre-SOFR"
    return "Post-SOFR"


def plot_figure2_public_decline(public: pd.DataFrame) -> Path:
    subset = public[
        public["group"].eq("g10")
        & public["tenor"].isin(["5y", "10y"])
        & public["cip_govt"].notna()
    ].copy()
    daily = (
        subset.groupby(["date", "tenor"], observed=True)["cip_govt"]
        .median()
        .reset_index()
        .sort_values("date")
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    colors = {"5y": "#1f77b4", "10y": "#d62728"}
    for tenor in ["5y", "10y"]:
        s = daily[daily["tenor"].eq(tenor)].set_index("date")["cip_govt"].rolling(21, min_periods=1).median()
        ax.plot(s.index, s.values, label=f"G10 median {tenor.upper()}", lw=1.6, color=colors[tenor])
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    markers = [
        ("2008-09-15", "Lehman"),
        ("2020-03-01", "COVID"),
        ("2022-01-01", "benchmark\ntransition"),
    ]
    for date, label in markers:
        ax.axvline(pd.Timestamp(date), color="#777777", ls="--", lw=1.0, alpha=0.75)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.92, label, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title("Replication Figure 2: Decline of U.S. Treasury Convenience (Public Processed Data)")
    ax.set_ylabel("Government bond CIP deviation / UST convenience (bps)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.text(0.01, 0.01, "Source: public Schreger CIP dataset v4. Series shown as 21-day rolling median of daily G10 medians.", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = REPORT_DIR / "replication_figure2_public_g10_median_5y_10y.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_figure3_term_structure(public: pd.DataFrame) -> tuple[Path, Path]:
    subset = public[
        public["group"].eq("g10")
        & public["tenor"].isin(TENOR_ORDER)
        & public["cip_govt"].notna()
    ].copy()
    subset["period"] = subset["date"].map(period_label)
    period_order = [
        "Pre-GFC",
        "Post-GFC / pre-COVID",
        "COVID shock",
        "Post-COVID / pre-SOFR",
        "Post-SOFR",
    ]
    summary = (
        subset.groupby(["period", "tenor"], observed=True)["cip_govt"]
        .median()
        .reset_index()
    )
    summary["tenor_years"] = summary["tenor"].map(TENOR_YEARS)
    summary = summary.dropna(subset=["tenor_years"])
    summary["period"] = pd.Categorical(summary["period"], categories=period_order, ordered=True)
    summary = summary.sort_values(["period", "tenor_years"])
    table_path = save_csv(summary, REPORT_DIR / "replication_figure3_public_term_structure_values.csv")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    colors = {
        "Pre-GFC": "#4C78A8",
        "Post-GFC / pre-COVID": "#F58518",
        "COVID shock": "#E45756",
        "Post-COVID / pre-SOFR": "#72B7B2",
        "Post-SOFR": "#54A24B",
    }
    for period in period_order:
        part = summary[summary["period"].astype(str).eq(period)]
        if part.empty:
            continue
        ax.plot(
            part["tenor_years"],
            part["cip_govt"],
            marker="o",
            lw=1.8,
            label=period,
            color=colors[period],
        )
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 2, 3, 5, 7, 10, 20, 30])
    ax.set_xticklabels(["3M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"])
    ax.set_title("Replication Figure 3: Term Structure of G10 UST Convenience")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Median government bond CIP deviation (bps)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.text(0.01, 0.01, "Source: public Schreger CIP dataset v4. Values are period medians across G10 observations.", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = REPORT_DIR / "replication_figure3_public_term_structure_g10.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, table_path


def make_table2_public(public: pd.DataFrame) -> tuple[Path, Path]:
    subset = public[public["tenor"].eq("5y") & public["cip_govt"].notna()].copy()
    subset["period"] = subset["date"].map(period_label)

    def summarize(keys: list[str]) -> pd.DataFrame:
        rows = (
            subset.groupby(keys, observed=True)
            .agg(
                n=("cip_govt", "count"),
                start_date=("date", "min"),
                end_date=("date", "max"),
                mean_bps=("cip_govt", "mean"),
                median_bps=("cip_govt", "median"),
                std_bps=("cip_govt", "std"),
                p25_bps=("cip_govt", lambda x: x.quantile(0.25)),
                p75_bps=("cip_govt", lambda x: x.quantile(0.75)),
                min_bps=("cip_govt", "min"),
                max_bps=("cip_govt", "max"),
            )
            .reset_index()
        )
        for col in ["mean_bps", "median_bps", "std_bps", "p25_bps", "p75_bps", "min_bps", "max_bps"]:
            rows[col] = rows[col].round(2)
        return rows

    group_summary = summarize(["group_label", "period"])
    major = subset[subset["currency"].isin(["EUR", "GBP", "JPY", "CHF", "CAD", "AUD"])]
    currency_summary = (
        major.groupby(["currency"], observed=True)
        .agg(
            n=("cip_govt", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            mean_bps=("cip_govt", "mean"),
            median_bps=("cip_govt", "median"),
            std_bps=("cip_govt", "std"),
            p25_bps=("cip_govt", lambda x: x.quantile(0.25)),
            p75_bps=("cip_govt", lambda x: x.quantile(0.75)),
            min_bps=("cip_govt", "min"),
            max_bps=("cip_govt", "max"),
        )
        .reset_index()
    )
    for col in ["mean_bps", "median_bps", "std_bps", "p25_bps", "p75_bps", "min_bps", "max_bps"]:
        currency_summary[col] = currency_summary[col].round(2)
    p1 = save_csv(group_summary, REPORT_DIR / "replication_table2_public_5y_summary.csv")
    p2 = save_csv(currency_summary, REPORT_DIR / "replication_table2_public_5y_major_currencies.csv")
    return p1, p2


def plot_distress_windows(public: pd.DataFrame) -> Path:
    subset = public[
        public["group"].eq("g10")
        & public["tenor"].eq("5y")
        & public["cip_govt"].notna()
    ].copy()
    daily = subset.groupby("date", observed=True)["cip_govt"].median().sort_index()
    windows = {
        "GFC / Lehman window": ("2008-07-01", "2009-03-31"),
        "COVID shock window": ("2020-02-01", "2020-06-30"),
        "Post-SOFR stress / repricing": ("2022-01-01", "2022-12-31"),
    }
    fig, axes = plt.subplots(len(windows), 1, figsize=(10.5, 8.2), sharey=False)
    for ax, (title, (start, end)) in zip(axes, windows.items()):
        s = daily.loc[pd.Timestamp(start) : pd.Timestamp(end)].rolling(5, min_periods=1).median()
        ax.plot(s.index, s.values, lw=1.6, color="#1f77b4")
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("bps")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Related Figure 8 Partial: G10 Government-CIP Stress Windows")
    fig.text(0.01, 0.01, "Source: public Schreger CIP dataset v4. Government-CIP-only partial replication; benchmark CIP not included.", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = REPORT_DIR / "replication_figure8_public_distress_windows.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_raw_pilot_latest_table_and_figure() -> tuple[Path | None, Path | None]:
    pilot_path = CLEANED_DIR / "bbg_pilot_cip_long.csv"
    public_path = CLEANED_DIR / "public_cip_cleaned.csv"
    if not pilot_path.exists() or not public_path.exists():
        return None, None
    pilot = pd.read_csv(pilot_path)
    pilot["date"] = pd.to_datetime(pilot["date"], errors="coerce")
    public = load_public()
    latest = (
        pilot.sort_values("date")
        .groupby(["currency", "tenor"], observed=True)
        .tail(1)
        .loc[:, ["currency", "tenor", "date", "rho_raw", "cip_govt_raw", "unit_convention_used"]]
        .sort_values(["currency", "tenor"])
    )
    latest["rho_raw"] = latest["rho_raw"].round(6)
    latest["cip_govt_raw"] = latest["cip_govt_raw"].round(3)
    table_path = save_csv(latest, REPORT_DIR / "raw_pilot_latest_values_may2026.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True)
    combos = [("EUR", "5y"), ("EUR", "10y"), ("GBP", "5y"), ("GBP", "10y")]
    for ax, (currency, tenor) in zip(axes.ravel(), combos):
        p = pilot[pilot["currency"].eq(currency) & pilot["tenor"].eq(tenor)].copy()
        pub = public[
            public["currency"].eq(currency)
            & public["tenor"].eq(tenor)
            & public["cip_govt"].notna()
        ].copy()
        pub = pub[pub["date"].ge(pd.Timestamp("2022-01-01"))]
        ax.plot(pub["date"], pub["cip_govt"], lw=1.2, label="Public processed", color="#ff7f0e")
        ax.plot(p["date"], p["cip_govt_raw"], lw=1.0, ls="--", label="Raw Bloomberg pilot", color="#1f77b4")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(f"{currency} {tenor.upper()}")
        ax.set_ylabel("bps")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelrotation=0)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Raw Bloomberg Pilot Extension vs Public Processed Data")
    fig.text(0.01, 0.01, "Raw pilot extends selected EUR/GBP 5Y/10Y series to May 29, 2026. Public processed data end at June 30, 2025.", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig_path = REPORT_DIR / "raw_public_latest_extension_5y_10y.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return table_path, fig_path


def make_data_needed_table(matrix: pd.DataFrame) -> Path:
    needs = matrix[
        ~matrix["public_processed_status"].eq("Ready")
        | ~matrix["current_raw_status"].str.startswith("Partial", na=False)
    ].copy()
    needs = needs[
        [
            "paper_item",
            "paper_title",
            "public_processed_status",
            "current_raw_status",
            "data_needed_for_full_raw_replication",
            "priority",
        ]
    ]
    return save_csv(needs, REPORT_DIR / "partial_replication_data_needed.csv")


def main() -> None:
    ensure_dirs()
    public = load_public()
    matrix = make_replication_matrix()
    paths = {
        "matrix": save_csv(matrix, REPORT_DIR / "figure_table_replication_matrix.csv"),
        "figure2": plot_figure2_public_decline(public),
    }
    fig3_path, fig3_table = plot_figure3_term_structure(public)
    t2_group, t2_currency = make_table2_public(public)
    paths.update(
        {
            "figure3": fig3_path,
            "figure3_values": fig3_table,
            "table2_group": t2_group,
            "table2_currency": t2_currency,
            "figure8_partial": plot_distress_windows(public),
            "data_needed": make_data_needed_table(matrix),
        }
    )
    raw_table, raw_fig = make_raw_pilot_latest_table_and_figure()
    if raw_table is not None:
        paths["raw_latest_table"] = raw_table
    if raw_fig is not None:
        paths["raw_latest_figure"] = raw_fig

    manifest = pd.DataFrame(
        [{"asset": key, "path": str(path.relative_to(PROJECT_ROOT))} for key, path in paths.items()]
    )
    save_csv(manifest, REPORT_DIR / "replication_report_asset_manifest.csv")
    print("Generated replication assets:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
