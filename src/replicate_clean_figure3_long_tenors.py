from __future__ import annotations

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
VALUES_LONG = CLEANED_DIR / "downloaded_bloomberg_values_long.csv"
PUBLIC_CIP = CLEANED_DIR / "public_cip_cleaned.csv"

COUNTRIES = ["CHF", "DKK", "EUR", "GBP", "JPY", "NOK", "SEK"]
TENORS = ["1y", "10y", "20y", "30y"]


def load_wide() -> pd.DataFrame:
    df = pd.read_csv(VALUES_LONG, usecols=["workbook", "date", "variable", "value"], parse_dates=["date"])
    df = df[df["workbook"].eq("figure3_g10_term")].copy()
    wide = df.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
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
    if ccy == "NOK":
        return first_col(columns, f"nok_common_irs_{tenor}_"), None
    low = ccy.lower()
    common = first_col(columns, f"{low}_common_irs_{tenor}_")
    prefix = f"{low}_{regime}_irs_{tenor}_"
    if ccy == "DKK" and regime == "ibor":
        if tenor in {"1y", "10y"} and common:
            return common, None
        if tenor == "30y":
            primary = first_col(columns, prefix, must_contain="v3")
            fallback = first_col(columns, prefix, exclude="v3")
            return primary or fallback, fallback if primary else None
        return first_col(columns, prefix, exclude="v3") or common, None
    if regime == "ibor":
        primary = first_col(columns, prefix, must_contain="v3")
        fallback = first_col(columns, prefix, exclude="v3")
        return primary or fallback or common, fallback if primary else None
    return first_col(columns, prefix) or common, None


def compute_panel(wide: pd.DataFrame) -> pd.DataFrame:
    cols = list(wide.columns)
    parts = []
    for ccy in COUNTRIES:
        low = ccy.lower()
        for tenor in TENORS:
            for regime in ["ibor", "sofr"]:
                basis = first_col(cols, f"{low}_{regime}_bs_{tenor}_")
                foreign_swap, swap_fallback = choose_swap(cols, ccy, regime, tenor)
                foreign_govt = first_col(cols, f"{low}_{regime}_govt_{tenor}_")
                usd_swap = first_col(cols, f"usd_{regime}_irs_{tenor}_")
                ust = first_col(cols, f"usd_{regime}_govt_{tenor}_")
                needed = [basis, foreign_swap, foreign_govt, usd_swap, ust]
                if any(v is None for v in needed):
                    print(f"WARNING missing {ccy} {tenor} {regime}: {needed}")
                    continue
                tbs = None
                tbs_sign = 0
                if ccy in {"JPY", "NOK"} and regime == "ibor":
                    tbs = first_col(cols, f"{low}_ibor_tbs_{tenor}_") or first_col(cols, f"{low}_common_tbs_{tenor}_")
                    tbs_sign = -1
                if ccy == "NOK" and regime == "sofr":
                    tbs = first_col(cols, f"{low}_common_tbs_{tenor}_")
                    tbs_sign = -1
                use = ["date", basis, foreign_swap, foreign_govt, usd_swap, ust]
                if swap_fallback:
                    use.append(swap_fallback)
                if tbs:
                    use.append(tbs)
                data = wide[use].copy()
                data = data.rename(
                    columns={
                        basis: "xccy_basis",
                        foreign_swap: "foreign_swap",
                        foreign_govt: "foreign_govt_yield",
                        usd_swap: "usd_swap",
                        ust: "ust_yield",
                    }
                )
                if swap_fallback:
                    data = data.rename(columns={swap_fallback: "foreign_swap_fallback"})
                    data["foreign_swap"] = data["foreign_swap"].combine_first(data["foreign_swap_fallback"])
                if tbs:
                    data = data.rename(columns={tbs: "tenor_basis"})
                numeric = ["xccy_basis", "foreign_swap", "foreign_govt_yield", "usd_swap", "ust_yield"]
                if tbs:
                    numeric.append("tenor_basis")
                data = data.dropna(subset=numeric)
                if data.empty:
                    continue
                tenor_adj = data["tenor_basis"] / 100.0 * tbs_sign if tbs else 0.0
                data["currency"] = ccy
                data["tenor"] = tenor
                data["regime"] = regime
                data["rho_raw"] = data["foreign_swap"] + data["xccy_basis"] / 100.0 + tenor_adj - data["usd_swap"]
                data["cip_govt_raw"] = 100.0 * (data["foreign_govt_yield"] - data["rho_raw"] - data["ust_yield"])
                data["basis_variable"] = basis
                data["swap_variable"] = foreign_swap
                data["swap_fallback_variable"] = swap_fallback or ""
                data["tenor_basis_variable"] = tbs or ""
                parts.append(data)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def combine_by_public_break(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    public = pd.read_csv(PUBLIC_CIP, parse_dates=["date", "cip_govt_break_date"])
    public = public[public["currency"].isin(COUNTRIES) & public["tenor"].astype(str).str.lower().isin(TENORS)].copy()
    parts = []
    for (ccy, tenor), group in public.groupby(["currency", "tenor"]):
        break_date = group["cip_govt_break_date"].dropna().iloc[0] if group["cip_govt_break_date"].notna().any() else pd.NaT
        old = raw[raw["currency"].eq(ccy) & raw["tenor"].eq(tenor) & raw["regime"].eq("ibor")]
        new = raw[raw["currency"].eq(ccy) & raw["tenor"].eq(tenor) & raw["regime"].eq("sofr")]
        if pd.isna(break_date):
            combined = old.copy()
        else:
            combined = pd.concat([old[old["date"].lt(break_date)], new[new["date"].ge(break_date)]], ignore_index=True)
        combined["break_date_used"] = break_date
        parts.append(combined)
    return pd.concat(parts, ignore_index=True), public


def compare(raw_combined: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ccy, tenor), r in raw_combined.groupby(["currency", "tenor"]):
        p = public[public["currency"].eq(ccy) & public["tenor"].astype(str).str.lower().eq(tenor)][["date", "cip_govt"]].dropna()
        m = r[["date", "cip_govt_raw"]].merge(p, on="date", how="inner").dropna()
        if m.empty:
            rows.append({"currency": ccy, "tenor": tenor, "n_overlap": 0, "status": "NO_OVERLAP"})
            continue
        diff = m["cip_govt_raw"] - m["cip_govt"]
        corr = m[["cip_govt_raw", "cip_govt"]].corr().iloc[0, 1]
        rows.append(
            {
                "currency": ccy,
                "tenor": tenor,
                "n_overlap": len(m),
                "date_min": m["date"].min(),
                "date_max": m["date"].max(),
                "corr": corr,
                "mae_bps": diff.abs().mean(),
                "mean_diff_bps": diff.mean(),
                "status": "GOOD_MATCH" if corr > 0.98 and diff.abs().mean() < 5 else "CHECK",
            }
        )
    return pd.DataFrame(rows)


def plot(raw_combined: pd.DataFrame, public: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = {"1y": "#cc0000", "10y": "#ffcc33", "20y": "#33aa33", "30y": "#222222"}
    for tenor in TENORS:
        s = (
            raw_combined[raw_combined["tenor"].eq(tenor)]
            .groupby("date")["cip_govt_raw"]
            .median()
            .sort_index()
            .rolling(13, min_periods=1)
            .mean()
        )
        ax.plot(s.index, s.values, color=colors[tenor], lw=1.1, label=f"Raw {tenor.upper()}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Figure 3 Partial Raw Replication: G10 ex AUD/CAD/NZD Long Tenors")
    ax.set_ylabel("Median government CIP deviation (bps)")
    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2026-06-30"))
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    out = FIGURE_DIR / "raw_figure3_g10_long_tenors_partial.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    wide = load_wide()
    raw = compute_panel(wide)
    combined, public = combine_by_public_break(raw)
    comp = compare(combined, public)
    fig = plot(combined, public)
    raw.to_csv(CLEANED_DIR / "raw_figure3_long_tenors_regime_panel.csv", index=False)
    combined.to_csv(CLEANED_DIR / "raw_figure3_long_tenors_combined_by_public_break.csv", index=False)
    comp.to_csv(TABLE_DIR / "raw_figure3_long_tenors_vs_public_comparison.csv", index=False)
    print("Raw Figure 3 long-tenor regime rows:", len(raw))
    print("Combined rows:", len(combined))
    print(comp.to_string(index=False))
    print("Figure:", fig)


if __name__ == "__main__":
    main()
