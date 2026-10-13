"""Public processed CIP replication figures with requested filenames."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / ".matplotlib"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from load_public_cip import load_public_cleaned
from utils import FIGURE_DIR


REFERENCE_LINES = [
    (pd.Timestamp("2008-09-15"), "GFC / Lehman"),
    (pd.Timestamp("2020-03-01"), "COVID shock"),
    (pd.Timestamp("2022-01-01"), "2022 benchmark transition"),
]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (11, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
        }
    )


def add_reference_lines(ax: plt.Axes) -> None:
    for date, label in REFERENCE_LINES:
        ax.axvline(date, color="0.35", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(
            date,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="0.25",
        )


def finish(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("cip_govt (basis points)")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.figure.autofmt_xdate()
    ax.text(
        0.01,
        -0.16,
        "Source: public Schreger CIP dataset v4. Processed-data replication.",
        transform=ax.transAxes,
        fontsize=8,
        color="0.35",
    )


def plot_public_g10_median(df: pd.DataFrame, tenor: str) -> str:
    subset = df[
        df["group"].astype(str).str.lower().eq("g10")
        & df["tenor"].astype(str).str.lower().eq(tenor)
    ].dropna(subset=["cip_govt"])
    series = subset.groupby("date")["cip_govt"].median().sort_index()
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, linewidth=1.3, label=f"G10 median {tenor.upper()}")
    add_reference_lines(ax)
    finish(ax, f"Public G10 Median {tenor.upper()} Government Bond CIP Deviation")
    ax.legend(frameon=False)
    out = FIGURE_DIR / f"public_g10_median_cip_govt_{tenor}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_public_major_currencies(df: pd.DataFrame, tenor: str) -> str:
    currencies = ["EUR", "GBP", "JPY"]
    subset = df[
        df["currency"].isin(currencies) & df["tenor"].astype(str).str.lower().eq(tenor)
    ].dropna(subset=["cip_govt"])
    fig, ax = plt.subplots()
    for currency in currencies:
        ccy_data = subset[subset["currency"].eq(currency)].sort_values("date")
        if ccy_data.empty:
            print(f"WARNING: Missing public {currency} {tenor.upper()} series.")
            continue
        ax.plot(ccy_data["date"], ccy_data["cip_govt"], linewidth=1.0, label=currency)
    add_reference_lines(ax)
    finish(ax, f"Public EUR / GBP / JPY {tenor.upper()} Government Bond CIP Deviation")
    ax.legend(frameon=False, ncol=3)
    out = FIGURE_DIR / f"public_eur_gbp_jpy_cip_govt_{tenor}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_public_g10_vs_em(df: pd.DataFrame, tenor: str) -> str | None:
    subset = df[df["tenor"].astype(str).str.lower().eq(tenor)].dropna(subset=["cip_govt"])
    if "group_label" not in subset.columns or subset["group_label"].nunique() < 2:
        print(f"WARNING: Cannot make G10 vs EM public plot for {tenor.upper()}.")
        return None
    med = (
        subset.groupby(["date", "group_label"])["cip_govt"]
        .median()
        .reset_index()
        .pivot(index="date", columns="group_label", values="cip_govt")
        .sort_index()
    )
    fig, ax = plt.subplots()
    for group in ["G10", "EM"]:
        if group in med.columns:
            ax.plot(med.index, med[group], linewidth=1.2, label=group)
    add_reference_lines(ax)
    finish(ax, f"Public G10 vs EM Median {tenor.upper()} Government Bond CIP Deviation")
    ax.legend(frameon=False)
    out = FIGURE_DIR / f"public_g10_vs_em_cip_govt_{tenor}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main() -> None:
    setup_matplotlib()
    df = load_public_cleaned()
    outputs = []
    for tenor in ["5y", "10y"]:
        outputs.append(plot_public_g10_median(df, tenor))
        outputs.append(plot_public_major_currencies(df, tenor))
        maybe = plot_public_g10_vs_em(df, tenor)
        if maybe:
            outputs.append(maybe)
    print("Generated public figures:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
