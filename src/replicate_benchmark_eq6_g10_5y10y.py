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

VALUES_LONG = CLEANED_DIR / "downloaded_bloomberg_values_long.csv"
PUBLIC_CIP = CLEANED_DIR / "public_cip_cleaned.csv"
COVERAGE = TABLE_DIR / "downloaded_bloomberg_ticker_coverage.csv"

G10 = ["AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"]
TENORS = ["5y", "10y"]
REGIMES = ["ibor", "sofr"]


def load_wide() -> pd.DataFrame:
    usecols = ["workbook", "date", "variable", "value"]
    values = pd.read_csv(VALUES_LONG, usecols=usecols, parse_dates=["date"])
    values = values[values["workbook"].eq("g10_benchmark_eq6")].copy()
    wide = values.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    wide.columns.name = None
    return wide.reset_index().sort_values("date")


def first_col(columns: list[str], prefix: str, must_contain: str | None = None, exclude: str | None = None) -> str | None:
    hits = [c for c in columns if c.startswith(prefix)]
    if must_contain:
        hits = [c for c in hits if must_contain in c]
    if exclude:
        hits = [c for c in hits if exclude not in c]
    return sorted(hits)[0] if hits else None


def choose_swap(columns: list[str], ccy: str, regime: str, tenor: str) -> tuple[str | None, str | None]:
    low = ccy.lower()
    prefix = f"{low}_{regime}_irs_{tenor}_"

    if ccy == "AUD":
        if regime == "sofr":
            return first_col(columns, prefix) or first_col(columns, f"{low}_ibor_irs_{tenor}_", must_contain="q"), None
        return first_col(columns, prefix, must_contain="q") or first_col(columns, prefix), None

    if ccy in {"NOK", "SEK"}:
        return first_col(columns, prefix) or first_col(columns, f"{low}_ibor_irs_{tenor}_"), None

    if ccy == "DKK":
        if regime == "sofr":
            return (
                first_col(columns, prefix)
                or first_col(columns, f"{low}_ibor_irs_{tenor}_", must_contain="v3")
                or first_col(columns, f"{low}_ibor_irs_{tenor}_")
            ), None
        primary = first_col(columns, prefix, must_contain="v3")
        fallback = first_col(columns, prefix, exclude="v3")
        return primary or fallback, fallback if primary else None

    if ccy in {"CHF", "EUR", "GBP"} and regime == "ibor":
        primary = first_col(columns, prefix, must_contain="v3")
        fallback = first_col(columns, prefix, exclude="v3")
        return primary or fallback, fallback if primary else None

    if ccy == "NZD" and regime == "ibor":
        return first_col(columns, prefix, exclude="blc") or first_col(columns, prefix), None

    return first_col(columns, prefix), None


def choose_tenor_basis(columns: list[str], ccy: str, regime: str, tenor: str) -> tuple[str | None, int]:
    low = ccy.lower()
    if ccy == "JPY" and regime == "sofr":
        # TONA IRS and TONA/SOFR basis already share the same floating benchmark.
        return None, 0
    if ccy in {"JPY", "NOK"}:
        tbs = first_col(columns, f"{low}_{regime}_tbs_{tenor}_") or first_col(columns, f"{low}_ibor_tbs_{tenor}_")
        return tbs, -1 if tbs else 0
    if ccy == "NZD" and regime == "sofr":
        tbs = first_col(columns, f"{low}_sofr_tbs_{tenor}_")
        return tbs, 1 if tbs else 0
    return None, 0


def compute_raw_panel(
    wide: pd.DataFrame,
    *,
    currencies: list[str] | None = None,
    regimes: list[str] | None = None,
) -> pd.DataFrame:
    columns = list(wide.columns)
    parts = []
    missing_rows = []

    for ccy in currencies if currencies is not None else G10:
        low = ccy.lower()
        for tenor in TENORS:
            for regime in regimes if regimes is not None else REGIMES:
                basis = first_col(columns, f"{low}_{regime}_bs_{tenor}_")
                foreign_swap, swap_fallback = choose_swap(columns, ccy, regime, tenor)
                foreign_govt = first_col(columns, f"{low}_{regime}_govt_{tenor}_")
                usd_swap = first_col(columns, f"usd_{regime}_irs_{tenor}_")
                usd_govt = first_col(columns, f"usd_{regime}_govt_{tenor}_")
                needed = [basis, foreign_swap, foreign_govt, usd_swap, usd_govt]
                if any(v is None for v in needed):
                    missing_rows.append(
                        {
                            "currency": ccy,
                            "tenor": tenor,
                            "regime": regime,
                            "basis": basis,
                            "foreign_swap": foreign_swap,
                            "foreign_govt": foreign_govt,
                            "usd_swap": usd_swap,
                            "usd_govt": usd_govt,
                        }
                    )
                    continue

                tenor_basis, tenor_basis_sign = choose_tenor_basis(columns, ccy, regime, tenor)
                usecols = ["date", basis, foreign_swap, foreign_govt, usd_swap, usd_govt]
                if swap_fallback:
                    usecols.append(swap_fallback)
                if tenor_basis:
                    usecols.append(tenor_basis)

                data = wide[usecols].copy()
                rename = {
                    basis: "xccy_basis",
                    foreign_swap: "foreign_swap",
                    foreign_govt: "foreign_govt_yield",
                    usd_swap: "usd_swap",
                    usd_govt: "usd_govt_yield",
                }
                if swap_fallback:
                    rename[swap_fallback] = "foreign_swap_fallback"
                if tenor_basis:
                    rename[tenor_basis] = "tenor_basis"
                data = data.rename(columns=rename)

                if swap_fallback:
                    data["foreign_swap"] = data["foreign_swap"].combine_first(data["foreign_swap_fallback"])

                numeric_cols = ["xccy_basis", "foreign_swap", "foreign_govt_yield", "usd_swap", "usd_govt_yield"]
                if tenor_basis:
                    numeric_cols.append("tenor_basis")
                data = data.dropna(subset=numeric_cols)
                if data.empty:
                    continue

                tenor_adjustment = data["tenor_basis"] / 100.0 * tenor_basis_sign if tenor_basis else 0.0
                data["currency"] = ccy
                data["tenor"] = tenor
                data["regime"] = regime
                data["basis_variable"] = basis
                data["swap_variable"] = foreign_swap
                data["swap_fallback_variable"] = swap_fallback or ""
                data["tenor_basis_variable"] = tenor_basis or ""
                data["tenor_basis_sign"] = tenor_basis_sign

                data["rho_raw"] = data["foreign_swap"] + data["xccy_basis"] / 100.0 + tenor_adjustment - data["usd_swap"]
                data["rho_raw_bps"] = 100.0 * data["rho_raw"]
                data["x_rf_raw_bps"] = 100.0 * (data["foreign_swap"] - data["rho_raw"] - data["usd_swap"])
                data["x_govt_raw_bps"] = 100.0 * (
                    data["foreign_govt_yield"] - data["rho_raw"] - data["usd_govt_yield"]
                )
                data["us_swap_spread_bps"] = 100.0 * (data["usd_swap"] - data["usd_govt_yield"])
                data["foreign_swap_spread_bps"] = 100.0 * (data["foreign_swap"] - data["foreign_govt_yield"])
                data["eq6_rhs_bps"] = (
                    data["x_rf_raw_bps"] + data["us_swap_spread_bps"] - data["foreign_swap_spread_bps"]
                )
                data["eq6_residual_bps"] = data["x_govt_raw_bps"] - data["eq6_rhs_bps"]
                parts.append(data)

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_missing_legs.csv", index=False)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mark_public_break_selection(panel: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["break_date_used"] = pd.NaT
    panel["is_selected_regime_by_public_break"] = False
    for (ccy, tenor), group in public.groupby(["currency", "tenor"]):
        break_dates = group["cip_govt_break_date"].dropna()
        break_date = break_dates.iloc[0] if not break_dates.empty else pd.NaT
        mask = panel["currency"].eq(ccy) & panel["tenor"].eq(str(tenor).lower())
        panel.loc[mask, "break_date_used"] = break_date
        if pd.isna(break_date):
            selected = mask & panel["regime"].eq("ibor")
        else:
            selected = mask & (
                (panel["regime"].eq("ibor") & panel["date"].lt(break_date))
                | (panel["regime"].eq("sofr") & panel["date"].ge(break_date))
            )
        panel.loc[selected, "is_selected_regime_by_public_break"] = True
    return panel


def comparison_stats(raw: pd.Series, public: pd.Series) -> dict[str, float | int | str]:
    merged = pd.concat([raw, public], axis=1, join="inner").dropna()
    if merged.empty:
        return {
            "n_overlap": 0,
            "date_min": "",
            "date_max": "",
            "corr": np.nan,
            "mae_bps": np.nan,
            "mean_diff_bps": np.nan,
            "max_abs_diff_bps": np.nan,
        }
    merged.columns = ["raw", "public"]
    diff = merged["raw"] - merged["public"]
    return {
        "n_overlap": int(len(merged)),
        "date_min": merged.index.min().date().isoformat(),
        "date_max": merged.index.max().date().isoformat(),
        "corr": float(merged[["raw", "public"]].corr().iloc[0, 1]),
        "mae_bps": float(diff.abs().mean()),
        "mean_diff_bps": float(diff.mean()),
        "max_abs_diff_bps": float(diff.abs().max()),
    }


def build_validation(panel: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    public = public.copy()
    public["tenor"] = public["tenor"].astype(str).str.lower()
    selected = panel[panel["is_selected_regime_by_public_break"]].copy()

    for (ccy, tenor), raw_group in selected.groupby(["currency", "tenor"]):
        public_group = public[public["currency"].eq(ccy) & public["tenor"].eq(tenor)].set_index("date")
        raw_indexed = raw_group.set_index("date")
        stats = comparison_stats(raw_indexed["x_govt_raw_bps"], public_group["cip_govt"])
        status = "GOOD_MATCH" if stats["n_overlap"] and stats["corr"] > 0.98 and stats["mae_bps"] < 3 else "CHECK"
        rows.append(
            {
                "check_type": "x_govt_vs_public_combined",
                "currency": ccy,
                "tenor": tenor,
                "regime": "selected_by_public_break",
                "raw_column": "x_govt_raw_bps",
                "public_column": "cip_govt",
                "status": status,
                **stats,
            }
        )
        stats = comparison_stats(raw_indexed["rho_raw_bps"], public_group["rho"] * 100.0)
        status = "DIAGNOSTIC_GOOD" if stats["n_overlap"] and stats["corr"] > 0.98 and stats["mae_bps"] < 3 else "DIAGNOSTIC_CHECK"
        rows.append(
            {
                "check_type": "rho_vs_public_combined",
                "currency": ccy,
                "tenor": tenor,
                "regime": "selected_by_public_break",
                "raw_column": "rho_raw_bps",
                "public_column": "rho_times_100",
                "status": status,
                **stats,
            }
        )

    for (ccy, tenor, regime), raw_group in panel.groupby(["currency", "tenor", "regime"]):
        public_group = public[public["currency"].eq(ccy) & public["tenor"].eq(tenor)].set_index("date")
        raw_indexed = raw_group.set_index("date")
        cip_col = f"cip_govt_{regime}"
        rho_col = f"rho_{regime}"
        if cip_col in public_group:
            stats = comparison_stats(raw_indexed["x_govt_raw_bps"], public_group[cip_col])
            status = "GOOD_MATCH" if stats["n_overlap"] and stats["corr"] > 0.98 and stats["mae_bps"] < 3 else "CHECK"
            rows.append(
                {
                    "check_type": "x_govt_vs_public_regime",
                    "currency": ccy,
                    "tenor": tenor,
                    "regime": regime,
                    "raw_column": "x_govt_raw_bps",
                    "public_column": cip_col,
                    "status": status,
                    **stats,
                }
            )
        if rho_col in public_group:
            stats = comparison_stats(raw_indexed["rho_raw_bps"], public_group[rho_col] * 100.0)
            status = (
                "DIAGNOSTIC_GOOD"
                if stats["n_overlap"] and stats["corr"] > 0.98 and stats["mae_bps"] < 3
                else "DIAGNOSTIC_CHECK"
            )
            rows.append(
                {
                    "check_type": "rho_vs_public_regime",
                    "currency": ccy,
                    "tenor": tenor,
                    "regime": regime,
                    "raw_column": "rho_raw_bps",
                    "public_column": f"{rho_col}_times_100",
                    "status": status,
                    **stats,
                }
            )

        residual = raw_group["eq6_residual_bps"].dropna()
        mae = float(residual.abs().mean()) if not residual.empty else np.nan
        max_abs = float(residual.abs().max()) if not residual.empty else np.nan
        rows.append(
            {
                "check_type": "eq6_identity_residual",
                "currency": ccy,
                "tenor": tenor,
                "regime": regime,
                "raw_column": "eq6_residual_bps",
                "public_column": "",
                "status": "IDENTITY_OK" if not residual.empty and mae < 1e-8 and max_abs < 1e-6 else "CHECK",
                "n_overlap": int(residual.notna().sum()),
                "date_min": raw_group["date"].min().date().isoformat(),
                "date_max": raw_group["date"].max().date().isoformat(),
                "corr": np.nan,
                "mae_bps": mae,
                "mean_diff_bps": float(residual.mean()) if not residual.empty else np.nan,
                "max_abs_diff_bps": max_abs,
            }
        )

    return pd.DataFrame(rows)


def build_component_summary(panel: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rho_raw_bps",
        "x_rf_raw_bps",
        "x_govt_raw_bps",
        "us_swap_spread_bps",
        "foreign_swap_spread_bps",
        "eq6_rhs_bps",
        "eq6_residual_bps",
    ]
    rows = []
    for (ccy, tenor, regime), group in panel.groupby(["currency", "tenor", "regime"]):
        row = {
            "currency": ccy,
            "tenor": tenor,
            "regime": regime,
            "n": int(len(group)),
            "date_min": group["date"].min().date().isoformat(),
            "date_max": group["date"].max().date().isoformat(),
            "selected_regime_n": int(group["is_selected_regime_by_public_break"].sum()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
        row["eq6_residual_abs_mean_bps"] = float(group["eq6_residual_bps"].abs().mean())
        row["eq6_residual_abs_max_bps"] = float(group["eq6_residual_bps"].abs().max())
        rows.append(row)
    return pd.DataFrame(rows)


def selected_median(panel: pd.DataFrame, tenor: str, column: str) -> pd.Series:
    selected = panel[panel["is_selected_regime_by_public_break"] & panel["tenor"].eq(tenor)]
    return selected.groupby("date")[column].median().sort_index().rolling(13, min_periods=1).mean()


def plot_benchmark_cip(panel: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    series = selected_median(panel, "5y", "x_rf_raw_bps")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(series.index, series.values, color="#2f6fbb", lw=1.2, label="G10 median 5Y benchmark CIP / x_rf")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Raw G10 5Y Benchmark CIP / Dollar-Convenience Proxy")
    ax.set_ylabel("Basis points")
    ax.set_xlabel("Date")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGURE_DIR / "raw_g10_5y_benchmark_cip_median.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_decomposition(panel: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    series = {
        "x_govt raw": selected_median(panel, "5y", "x_govt_raw_bps"),
        "x_rf raw": selected_median(panel, "5y", "x_rf_raw_bps"),
        "US swap spread": selected_median(panel, "5y", "us_swap_spread_bps"),
        "- foreign swap spread": -selected_median(panel, "5y", "foreign_swap_spread_bps"),
        "Eq.6 RHS": selected_median(panel, "5y", "eq6_rhs_bps"),
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = {
        "x_govt raw": "#222222",
        "x_rf raw": "#2f6fbb",
        "US swap spread": "#2b8c4a",
        "- foreign swap spread": "#d97925",
        "Eq.6 RHS": "#9a4fb4",
    }
    for label, s in series.items():
        style = "--" if label == "Eq.6 RHS" else "-"
        ax.plot(s.index, s.values, lw=1.1, color=colors[label], ls=style, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Raw G10 5Y Equation 6 Decomposition Components")
    ax.set_ylabel("Basis points")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out = FIGURE_DIR / "raw_g10_5y_eq6_decomposition_median.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pd.read_csv(COVERAGE, nrows=1)
    wide = load_wide()
    public = pd.read_csv(PUBLIC_CIP, parse_dates=["date", "cip_govt_break_date"])
    public = public[public["currency"].isin(G10) & public["tenor"].astype(str).str.lower().isin(TENORS)].copy()
    public["tenor"] = public["tenor"].astype(str).str.lower()

    panel = compute_raw_panel(wide)
    panel = mark_public_break_selection(panel, public)
    validation = build_validation(panel, public)
    component_summary = build_component_summary(panel)
    benchmark_fig = plot_benchmark_cip(panel)
    decomposition_fig = plot_decomposition(panel)

    panel.to_csv(CLEANED_DIR / "raw_g10_benchmark_eq6_5y10y_panel.csv", index=False)
    validation.to_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_validation.csv", index=False)
    component_summary.to_csv(TABLE_DIR / "raw_g10_benchmark_eq6_5y10y_component_summary.csv", index=False)

    identity = validation[validation["check_type"].eq("eq6_identity_residual")]
    govt = validation[validation["check_type"].eq("x_govt_vs_public_combined")]
    print("Raw G10 benchmark / Eq.6 panel rows:", len(panel))
    print("Eq.6 residual MAE max:", identity["mae_bps"].max())
    print("Combined government-CIP checks:", govt["status"].value_counts().to_dict())
    print("Benchmark CIP figure:", benchmark_fig)
    print("Equation 6 decomposition figure:", decomposition_fig)


if __name__ == "__main__":
    main()
