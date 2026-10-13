"""Identity and benchmark-transition checks for the processed CIP data."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / ".matplotlib"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from load_and_clean import g10_filter, load_cleaned_dataset, tenor_filter


FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"


def identity_check(df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    needed = ["diff_y", "rho", "cip_govt"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        out = TABLE_DIR / "identity_check.csv"
        pd.DataFrame({"status": [f"Missing columns: {missing}"]}).to_csv(out, index=False)
        return out

    subset = df.dropna(subset=needed).copy()
    candidates = {
        "diff_y - rho": subset["diff_y"] - subset["rho"],
        "rho - diff_y": subset["rho"] - subset["diff_y"],
        "100 * (diff_y - rho)": 100.0 * (subset["diff_y"] - subset["rho"]),
        "100 * (rho - diff_y)": 100.0 * (subset["rho"] - subset["diff_y"]),
        "(diff_y - rho) / 100": (subset["diff_y"] - subset["rho"]) / 100.0,
        "(rho - diff_y) / 100": (subset["rho"] - subset["diff_y"]) / 100.0,
    }

    rows = []
    for label, candidate in candidates.items():
        err = (candidate - subset["cip_govt"]).dropna()
        rows.append(
            {
                "candidate_identity": f"cip_govt = {label}",
                "n": int(err.shape[0]),
                "mean_absolute_error": float(err.abs().mean()),
                "median_absolute_error": float(err.abs().median()),
                "max_absolute_error": float(err.abs().max()),
            }
        )
    result = pd.DataFrame(rows)
    result["best_match"] = result["mean_absolute_error"].eq(result["mean_absolute_error"].min())
    # Values are rounded in the public CSV, so use a small basis-point tolerance.
    result["holds_approximately"] = result["mean_absolute_error"].lt(1e-3)
    out = TABLE_DIR / "identity_check.csv"
    result.to_csv(out, index=False)
    result.to_csv(TABLE_DIR / "public_identity_check.csv", index=False)
    return out


def benchmark_coverage(df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_cols = [
        "rho_ibor",
        "rho_sofr",
        "cip_govt_ibor",
        "cip_govt_sofr",
        "cip_govt_break_date",
    ]
    rows = []
    for col in benchmark_cols:
        if col in df.columns:
            rows.append(
                {
                    "column": col,
                    "non_missing": int(df[col].notna().sum()),
                    "missing": int(df[col].isna().sum()),
                    "coverage_share": float(df[col].notna().mean()),
                }
            )
        else:
            rows.append({"column": col, "non_missing": 0, "missing": len(df), "coverage_share": 0.0})
    out = TABLE_DIR / "benchmark_variable_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def benchmark_transition_summary(df: pd.DataFrame) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if "cip_govt_break_date" not in df.columns:
        out = TABLE_DIR / "benchmark_transition_summary.csv"
        pd.DataFrame({"status": ["cip_govt_break_date not available"]}).to_csv(out, index=False)
        return out

    rows = []
    for currency, group in df.groupby("currency", observed=True):
        row = {
            "currency": currency,
            "group": group["group_label"].dropna().mode().iloc[0]
            if not group["group_label"].dropna().empty
            else None,
            "break_dates": "; ".join(
                pd.Timestamp(d).strftime("%Y-%m-%d")
                for d in sorted(group["cip_govt_break_date"].dropna().unique())
            ),
        }
        for col in ["rho_ibor", "rho_sofr", "cip_govt_ibor", "cip_govt_sofr"]:
            if col in group.columns:
                row[f"{col}_non_missing"] = int(group[col].notna().sum())
        rows.append(row)

    out = TABLE_DIR / "benchmark_transition_summary.csv"
    pd.DataFrame(rows).sort_values(["group", "currency"]).to_csv(out, index=False)
    return out


def plot_ibor_sofr(df: pd.DataFrame, tenor: str) -> Path | None:
    cols = [col for col in ["cip_govt", "cip_govt_ibor", "cip_govt_sofr"] if col in df.columns]
    if len(cols) < 2:
        print(f"WARNING: Not enough benchmark columns for {tenor.upper()} IBOR/SOFR comparison.")
        return None

    subset = df[g10_filter(df) & tenor_filter(df, tenor)]
    med = subset.groupby("date")[cols].median().sort_index()
    if med.dropna(how="all").empty:
        print(f"WARNING: Empty benchmark comparison data for {tenor.upper()}.")
        return None

    labels = {
        "cip_govt": "Main cip_govt",
        "cip_govt_ibor": "IBOR version",
        "cip_govt_sofr": "SOFR / alt benchmark version",
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in cols:
        ax.plot(med.index, med[col], linewidth=1.2, label=labels.get(col, col))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.axvline(pd.Timestamp("2022-01-01"), color="0.35", linestyle="--", linewidth=0.9)
    ax.set_title(f"G10 Median {tenor.upper()} CIP: Main vs IBOR/SOFR Processed Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("cip_govt (basis points)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    ax.text(
        0.01,
        -0.16,
        "Source: public Schreger CIP dataset v4. Median across G10 currencies.",
        transform=ax.transAxes,
        fontsize=8,
        color="0.35",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"ibor_sofr_comparison_{tenor}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    df = load_cleaned_dataset()
    outputs: list[Path] = [
        identity_check(df),
        benchmark_coverage(df),
        benchmark_transition_summary(df),
    ]
    for tenor in ["5y", "10y"]:
        maybe = plot_ibor_sofr(df, tenor)
        if maybe is not None:
            outputs.append(maybe)
    print("Generated identity / benchmark outputs:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
