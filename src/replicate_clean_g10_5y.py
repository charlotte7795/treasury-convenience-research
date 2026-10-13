from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"
VALUES_LONG = CLEANED_DIR / "downloaded_bloomberg_values_long.csv"
PUBLIC_CIP = CLEANED_DIR / "public_cip_cleaned.csv"

G10 = ["AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"]


@dataclass(frozen=True)
class FormulaLegs:
    currency: str
    regime: str
    xccy_basis: str
    foreign_swap: str
    foreign_govt_yield: str
    usd_swap: str
    ust_yield: str
    public_col: str
    foreign_swap_fallback: str | None = None
    tenor_basis: str | None = None
    tenor_basis_sign: int = 0


FORMULA_LEGS = [
    FormulaLegs("AUD", "ibor", "aud_ibor_xccy_basis_5y", "aud_ibor_swap_qtr_5y", "aud_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor"),
    FormulaLegs("CAD", "ibor", "cad_ibor_xccy_basis_5y", "cad_ibor_swap_5y", "cad_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor"),
    FormulaLegs("CHF", "ibor", "chf_ibor_xccy_basis_5y", "chf_ibor_swap_3m_5y", "chf_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor", "chf_ibor_swap_6m_precutoff_5y"),
    FormulaLegs("DKK", "ibor", "dkk_ibor_xccy_basis_5y", "dkk_ibor_swap_3m_alt_5y", "dkk_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor"),
    FormulaLegs("EUR", "ibor", "eur_ibor_xccy_basis_5y", "eur_ibor_swap_3m_5y", "eur_germany_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor", "eur_ibor_swap_6m_precutoff_5y"),
    FormulaLegs("GBP", "ibor", "gbp_ibor_xccy_basis_5y", "gbp_ibor_swap_3m_5y", "gbp_gilt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor", "gbp_ibor_swap_6m_precutoff_5y"),
    FormulaLegs("JPY", "ibor", "jpy_ibor_xccy_basis_5y", "jpy_ibor_swap_5y", "jpy_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor", tenor_basis="jpy_tenor_basis_3v6_5y", tenor_basis_sign=-1),
    FormulaLegs("NOK", "ibor", "nok_ibor_xccy_basis_5y", "nok_ibor_swap_5y", "nok_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor", tenor_basis="nok_tenor_basis_3v6_5y", tenor_basis_sign=-1),
    FormulaLegs("NZD", "ibor", "nzd_ibor_xccy_basis_5y", "nzd_ibor_swap_5y", "nzd_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor", "nzd_ibor_swap_alt_5y"),
    FormulaLegs("SEK", "ibor", "sek_ibor_xccy_basis_5y", "sek_ibor_swap_5y", "sek_govt_old_5y", "usd_ibor_swap_5y", "ust_old_curve_5y_backup", "cip_govt_ibor"),
    FormulaLegs("AUD", "sofr", "aud_sofr_xccy_basis_5y", "aud_sofr_swap_5y", "aud_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
    FormulaLegs("CAD", "sofr", "cad_sofr_xccy_basis_5y", "cad_sofr_swap_5y", "cad_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
    FormulaLegs("CHF", "sofr", "chf_sofr_xccy_basis_5y", "chf_saron_swap_5y", "chf_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
    FormulaLegs("DKK", "sofr", "dkk_sofr_xccy_basis_5y", "dkk_ibor_swap_3m_5y", "dkk_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
    FormulaLegs("EUR", "sofr", "eur_sofr_xccy_basis_5y", "eur_estr_swap_5y", "eur_germany_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
    FormulaLegs("GBP", "sofr", "gbp_sofr_xccy_basis_5y", "gbp_sonia_swap_5y", "gbp_gilt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
    FormulaLegs("JPY", "sofr", "jpy_sofr_xccy_basis_5y", "jpy_tona_swap_5y", "jpy_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
    FormulaLegs("NOK", "sofr", "nok_sofr_xccy_basis_5y", "nok_ibor_swap_5y", "nok_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr", tenor_basis="nok_tenor_basis_3v6_5y", tenor_basis_sign=-1),
    FormulaLegs("NZD", "sofr", "nzd_sofr_xccy_basis_5y", "nzd_sofr_swap_5y", "nzd_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr", tenor_basis="nzd_tenor_basis_5y", tenor_basis_sign=1),
    FormulaLegs("SEK", "sofr", "sek_sofr_xccy_basis_5y", "sek_ibor_swap_5y", "sek_govt_bval_5y", "usd_sofr_swap_5y", "ust_bval_5y", "cip_govt_sofr"),
]


def load_downloaded_wide(workbook: str) -> pd.DataFrame:
    usecols = ["workbook", "date", "variable", "value"]
    df = pd.read_csv(VALUES_LONG, usecols=usecols, parse_dates=["date"])
    df = df[df["workbook"].eq(workbook)].copy()
    wide = df.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    wide.columns.name = None
    return wide.reset_index().sort_values("date")


def compute_raw_panel(wide: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for legs in FORMULA_LEGS:
        needed = [legs.xccy_basis, legs.foreign_swap, legs.foreign_govt_yield, legs.usd_swap, legs.ust_yield]
        if legs.foreign_swap_fallback:
            needed.append(legs.foreign_swap_fallback)
        if legs.tenor_basis:
            needed.append(legs.tenor_basis)
        missing = [c for c in needed if c not in wide.columns]
        if missing:
            print(f"WARNING: missing {legs.currency} {legs.regime}: {missing}")
            continue
        data = wide[["date"] + needed].copy()
        rename = {
            legs.xccy_basis: "xccy_basis",
            legs.foreign_swap: "foreign_swap",
            legs.foreign_govt_yield: "foreign_govt_yield",
            legs.usd_swap: "usd_swap",
            legs.ust_yield: "ust_yield",
        }
        if legs.foreign_swap_fallback:
            rename[legs.foreign_swap_fallback] = "foreign_swap_fallback"
        if legs.tenor_basis:
            rename[legs.tenor_basis] = "tenor_basis"
        data = data.rename(columns=rename)
        if legs.foreign_swap_fallback:
            data["foreign_swap"] = data["foreign_swap"].combine_first(data["foreign_swap_fallback"])
        numeric_cols = ["xccy_basis", "foreign_swap", "foreign_govt_yield", "usd_swap", "ust_yield"]
        if legs.tenor_basis:
            numeric_cols.append("tenor_basis")
        data = data.dropna(subset=numeric_cols)
        if data.empty:
            continue
        tenor_adj = data["tenor_basis"] / 100.0 * legs.tenor_basis_sign if legs.tenor_basis else 0.0
        data["currency"] = legs.currency
        data["tenor"] = "5y"
        data["regime"] = legs.regime
        data["rho_raw"] = data["foreign_swap"] + data["xccy_basis"] / 100.0 + tenor_adj - data["usd_swap"]
        data["cip_govt_raw"] = 100.0 * (data["foreign_govt_yield"] - data["rho_raw"] - data["ust_yield"])
        data["public_col"] = legs.public_col
        data["tenor_basis_adjustment"] = legs.tenor_basis or ""
        data["tenor_basis_sign"] = legs.tenor_basis_sign
        parts.append(data)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_public_5y() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC_CIP, parse_dates=["date", "cip_govt_break_date"])
    return public[public["tenor"].astype(str).str.lower().eq("5y") & public["currency"].isin(G10)].copy()


def combine_by_break_date(raw: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    breaks = public.groupby("currency")["cip_govt_break_date"].first().to_dict()
    ibor = raw[raw["regime"].eq("ibor")].copy()
    sofr = raw[raw["regime"].eq("sofr")].copy()
    parts = []
    for ccy in G10:
        break_date = breaks.get(ccy)
        old = ibor[ibor["currency"].eq(ccy)]
        new = sofr[sofr["currency"].eq(ccy)]
        if pd.isna(break_date):
            combined = old.copy()
        else:
            combined = pd.concat([old[old["date"].lt(break_date)], new[new["date"].ge(break_date)]], ignore_index=True)
        combined["break_date_used"] = break_date
        parts.append(combined)
    return pd.concat(parts, ignore_index=True).sort_values(["currency", "date"])


def comparison_table(raw: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ccy in G10:
        r = raw[raw["currency"].eq(ccy)][["date", "cip_govt_raw"]]
        p = public[public["currency"].eq(ccy)][["date", "cip_govt"]].dropna()
        merged = r.merge(p, on="date", how="inner").dropna()
        if merged.empty:
            rows.append({"currency": ccy, "n_overlap": 0, "status": "NO_OVERLAP"})
            continue
        diff = merged["cip_govt_raw"] - merged["cip_govt"]
        rows.append(
            {
                "currency": ccy,
                "n_overlap": len(merged),
                "date_min": merged["date"].min(),
                "date_max": merged["date"].max(),
                "corr": merged[["cip_govt_raw", "cip_govt"]].corr().iloc[0, 1],
                "mae_bps": diff.abs().mean(),
                "mean_diff_bps": diff.mean(),
                "max_abs_diff_bps": diff.abs().max(),
                "status": "GOOD_MATCH" if merged[["cip_govt_raw", "cip_govt"]].corr().iloc[0, 1] > 0.98 and diff.abs().mean() < 3 else "CHECK",
            }
        )
    return pd.DataFrame(rows)


def nw_se_mean(x: pd.Series, lags: int = 91) -> float:
    y = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 2:
        return float("nan")
    y = y - y.mean()
    gamma0 = float(np.dot(y, y) / n)
    lrv = gamma0
    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(y[lag:], y[:-lag]) / n)
        lrv += 2.0 * weight * gamma
    return float(np.sqrt(max(lrv, 0.0) / n))


def table2_stats(raw_combined: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("Full Sample", "2000-01-01", "2025-06-30"),
        ("2000-2006", "2000-01-01", "2006-12-31"),
        ("2010-2020", "2010-01-01", "2020-12-31"),
        ("2021-2025", "2021-01-01", "2025-06-30"),
    ]
    rows = []
    for ccy in G10:
        sub = raw_combined[raw_combined["currency"].eq(ccy)]
        for label, start, end in periods:
            s = sub[sub["date"].between(pd.Timestamp(start), pd.Timestamp(end))]["cip_govt_raw"]
            rows.append(
                {
                    "currency": ccy,
                    "period": label,
                    "mean_bps": s.mean(),
                    "nw_se_bps_lag91": nw_se_mean(s, 91),
                    "n": int(s.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_g10_figure2(raw_combined: pd.DataFrame, public: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_daily = raw_combined.groupby("date")["cip_govt_raw"].median().sort_index().rolling(13, min_periods=1).mean()
    public_daily = public.groupby("date")["cip_govt"].median().sort_index().rolling(13, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(public_daily.index, public_daily.values, color="#999999", lw=1.1, label="Public processed G10 median 5Y")
    ax.plot(raw_daily.index, raw_daily.values, color="#cc0000", lw=1.1, label="Raw Bloomberg reconstruction G10 median 5Y")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2026-06-30"))
    ax.set_ylabel("Government bond CIP deviation / UST convenience (bps)")
    ax.set_title("G10 5Y Government CIP: Raw Bloomberg Reconstruction vs Public Processed")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGURE_DIR / "raw_g10_5y_figure2_g10_public_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    wide = load_downloaded_wide("g10_5y")
    raw = compute_raw_panel(wide)
    public = load_public_5y()
    combined = combine_by_break_date(raw, public)
    comp = comparison_table(combined, public)
    table2 = table2_stats(combined)
    fig = plot_g10_figure2(combined, public)

    raw.to_csv(CLEANED_DIR / "raw_g10_5y_regime_panel.csv", index=False)
    combined.to_csv(CLEANED_DIR / "raw_g10_5y_combined_by_public_break.csv", index=False)
    comp.to_csv(TABLE_DIR / "raw_g10_5y_vs_public_comparison.csv", index=False)
    table2.to_csv(TABLE_DIR / "raw_g10_5y_table2_stats.csv", index=False)

    print("Raw G10 5Y regime rows:", len(raw))
    print("Raw G10 5Y combined rows:", len(combined))
    print("Comparison:")
    print(comp.to_string(index=False))
    print("Table 2 stats:", TABLE_DIR / "raw_g10_5y_table2_stats.csv")
    print("Figure:", fig)


if __name__ == "__main__":
    main()
