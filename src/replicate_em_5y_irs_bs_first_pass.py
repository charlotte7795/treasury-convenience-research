from __future__ import annotations

import os
from pathlib import Path

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


def load_wide() -> pd.DataFrame:
    df = pd.read_csv(VALUES_LONG, usecols=["workbook", "date", "variable", "value"], parse_dates=["date"])
    df = df[df["workbook"].eq("em_5y")].copy()
    wide = df.pivot_table(index="date", columns="variable", values="value", aggfunc="first")
    wide.columns.name = None
    return wide.reset_index().sort_values("date")


def first(columns: list[str], prefix: str, prefer_non_bval: bool = False) -> str | None:
    hits = sorted([c for c in columns if c.startswith(prefix)])
    if prefer_non_bval:
        non_bval = [c for c in hits if "bv" not in c.lower()]
        if non_bval:
            return non_bval[0]
    return hits[0] if hits else None


def candidate_currencies(wide: pd.DataFrame) -> list[str]:
    cols = list(wide.columns)
    ccys = sorted({c.split("_", 1)[0].upper() for c in cols if "_ibor_bs_5y_" in c})
    out = []
    for ccy in ccys:
        low = ccy.lower()
        if (
            first(cols, f"{low}_ibor_bs_5y_")
            and first(cols, f"{low}_ibor_irs_5y_")
            and first(cols, f"{low}_ibor_govt_5y_", prefer_non_bval=True)
        ):
            out.append(ccy)
    return out


def diagnostic_status(corr: float, mae_bps: float, n_overlap: int) -> str:
    if n_overlap > 1000 and corr > 0.99 and mae_bps < 3:
        return "VALIDATED"
    if n_overlap > 1000 and corr > 0.98 and mae_bps <= 12:
        return "USABLE_DIAGNOSTIC"
    return "REVIEW"


def build_first_pass() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = load_wide()
    public = pd.read_csv(PUBLIC_CIP, parse_dates=["date"])
    cols = list(wide.columns)
    usd_swap = first(cols, "usd_ibor_irs_5y_")
    ust = first(cols, "usd_ibor_govt_5y_")
    if not usd_swap or not ust:
        raise ValueError("Missing USD 5Y IBOR swap or Treasury leg in EM workbook.")

    panels = []
    comparisons = []
    for ccy in candidate_currencies(wide):
        low = ccy.lower()
        basis = first(cols, f"{low}_ibor_bs_5y_")
        swap = first(cols, f"{low}_ibor_irs_5y_")
        govt = first(cols, f"{low}_ibor_govt_5y_", prefer_non_bval=True)
        data = wide[["date", basis, swap, govt, usd_swap, ust]].dropna().copy()
        if data.empty:
            continue
        pub = public[
            public["currency"].eq(ccy)
            & public["tenor"].astype(str).str.lower().eq("5y")
            & public["cip_govt_ibor"].notna()
        ][["date", "cip_govt_ibor"]]
        best = None
        best_panel = None
        for sign in [1, -1]:
            rho = data[swap] + sign * data[basis] / 100.0 - data[usd_swap]
            cip = 100.0 * (data[govt] - rho - data[ust])
            panel = data[["date"]].copy()
            panel["currency"] = ccy
            panel["regime"] = "ibor"
            panel["tenor"] = "5y"
            panel["basis_sign"] = sign
            panel["rho_raw"] = rho
            panel["cip_govt_raw"] = cip
            merged = panel[["date", "cip_govt_raw"]].merge(pub, on="date", how="inner").dropna()
            if merged.empty:
                continue
            diff = merged["cip_govt_raw"] - merged["cip_govt_ibor"]
            corr = merged[["cip_govt_raw", "cip_govt_ibor"]].corr().iloc[0, 1]
            mae_bps = diff.abs().mean()
            row = {
                "currency": ccy,
                "basis_sign": sign,
                "n_overlap": len(merged),
                "date_min": merged["date"].min(),
                "date_max": merged["date"].max(),
                "corr": corr,
                "mae_bps": mae_bps,
                "mean_diff_bps": diff.mean(),
                "basis_variable": basis,
                "swap_variable": swap,
                "govt_variable": govt,
                "status": diagnostic_status(corr, mae_bps, len(merged)),
            }
            if best is None or row["mae_bps"] < best["mae_bps"]:
                best = row
                best_panel = panel
        if best is not None and best_panel is not None:
            comparisons.append(best)
            panels.append(best_panel)

    panel_out = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    comp_out = pd.DataFrame(comparisons).sort_values(["status", "currency"]) if comparisons else pd.DataFrame()
    return panel_out, comp_out


def plot(panel: pd.DataFrame) -> Path | None:
    if panel.empty:
        return None
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_daily = panel.groupby("date")["cip_govt_raw"].median().sort_index().rolling(13, min_periods=1).mean()
    public = pd.read_csv(PUBLIC_CIP, parse_dates=["date"])
    public_daily = (
        public[
            public["group"].astype(str).str.upper().eq("EM")
            & public["tenor"].astype(str).str.lower().eq("5y")
            & public["cip_govt"].notna()
        ]
        .groupby("date")["cip_govt"]
        .median()
        .sort_index()
        .rolling(13, min_periods=1)
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(public_daily.index, public_daily.values, color="#999999", lw=1.0, label="Public EM median 5Y")
    ax.plot(raw_daily.index, raw_daily.values, color="#1f77b4", lw=1.0, label="Raw EM IRS+BS diagnostic subset")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("EM 5Y Diagnostic First Pass: IRS + Basis Swap Subset")
    ax.set_ylabel("Government bond CIP deviation (bps)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGURE_DIR / "raw_em_5y_irs_bs_diagnostic_subset.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    panel, comp = build_first_pass()
    fig = plot(panel)
    panel.to_csv(CLEANED_DIR / "raw_em_5y_irs_bs_diagnostic_panel.csv", index=False)
    comp.to_csv(TABLE_DIR / "raw_em_5y_irs_bs_diagnostic_comparison.csv", index=False)
    print(comp.to_string(index=False))
    print("Panel:", CLEANED_DIR / "raw_em_5y_irs_bs_diagnostic_panel.csv")
    print("Comparison:", TABLE_DIR / "raw_em_5y_irs_bs_diagnostic_comparison.csv")
    print("Figure:", fig)


if __name__ == "__main__":
    main()
