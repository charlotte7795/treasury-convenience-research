"""Retain exact-ticker provenance while refreshing the fixed-eight 5Y/10Y study."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.analyze_treasury_interim import CLEAN, COHORT, ROOT, TABLE
from src.replicate_benchmark_eq6_g10_5y10y import (
    build_validation,
    compute_raw_panel,
    mark_public_break_selection,
)

REFRESH = ROOT / "data" / "refresh" / "Foreign_Spread_5Y10Y_2026-10-09.xlsx"
DOWNLOAD = Path.home() / "Downloads" / "Foreign_Spread_5Y10Y_Refresh_Bloomberg_BDH.xlsx"
INPUTS = TABLE / "foreign_swap_spread_refresh_inputs.csv"
PANEL = CLEAN / "raw_g10_benchmark_eq6_5y10y_refreshed_panel.csv"


def read_refresh(path: Path, expected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, profiles = [], []
    wb = load_workbook(path, read_only=True, data_only=True)
    for sheet in ["USD", *COHORT]:
        rows = list(wb[sheet].iter_rows(values_only=True))
        for col in range(0, len(rows[0]), 3):
            if col + 1 >= len(rows[0]) or not rows[0][col + 1]:
                continue
            ticker, variable, field, unit = [str(row[col + 1]) for row in rows[:4]]
            dates, quotes, errors = [], [], 0
            for row in rows[4:]:
                dt, value = row[col:col + 2]
                if dt is None and value is None:
                    continue
                if not isinstance(dt, (datetime, date)) or not isinstance(value, (int, float)):
                    errors += 1
                    continue
                if isinstance(value, bool) or not np.isfinite(value):
                    errors += 1
                    continue
                dates.append(pd.Timestamp(dt))
                quotes.append(float(value))
            series = pd.DataFrame({"date": dates, "value": quotes})
            assert not series.empty, f"Empty refresh block: {ticker}"
            assert errors == 0, f"Invalid refresh cells: {ticker}"
            assert not series.date.duplicated().any(), f"Duplicate refresh dates: {ticker}"
            assert series.date.is_monotonic_increasing
            assert series.date.dt.weekday.lt(5).all(), f"Weekend quotes: {ticker}"
            meta = expected.set_index("variable").loc[variable]
            assert ticker.upper() == meta.ticker.upper()
            assert field == meta.field and unit == meta.unit
            profiles.append({"currency": sheet, "variable": variable, "ticker": ticker,
                             "n_numeric": len(series), "first_date": series.date.min(),
                             "last_date": series.date.max(), "errors": errors,
                             "min": series.value.min(), "max": series.value.max(), "unit": unit})
            series["ticker"] = ticker.upper()
            series["variable"] = variable
            values.append(series)
    wb.close()
    values, profiles = pd.concat(values, ignore_index=True), pd.DataFrame(profiles)
    assert set(profiles.variable) == set(expected.variable)
    assert len(profiles) == len(expected) == 54
    return values, profiles


def exact_history(expected: pd.DataFrame) -> pd.DataFrame:
    parts = []
    mapping = expected[["variable", "ticker"]].copy()
    mapping["ticker"] = mapping.ticker.str.upper()
    for chunk in pd.read_csv(CLEAN / "downloaded_bloomberg_values_long.csv",
                             usecols=["workbook", "date", "variable", "ticker", "value"],
                             chunksize=250_000):
        sub = chunk[chunk.workbook.eq("g10_benchmark_eq6") & chunk.variable.isin(expected.variable)
                    & chunk.date.ge("2023-01-01")].copy()
        sub["ticker"] = sub.ticker.str.upper()
        parts.append(sub.merge(mapping, on=["variable", "ticker"], validate="many_to_one"))
    history = pd.concat(parts, ignore_index=True)
    history["date"] = pd.to_datetime(history.date)
    check = history.groupby(["date", "variable", "ticker"]).value.agg(["min", "max"])
    assert (check["max"] - check["min"]).lt(1e-9).all(), "Conflicting exact-ticker history"
    return history.drop_duplicates(["date", "variable", "ticker"])


def retain_observed_history(new: pd.DataFrame, old: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["date", "variable", "ticker"]
    retained = old.merge(new[keys], on=keys, how="left", indicator=True, validate="one_to_one")
    retained = retained[retained._merge.eq("left_only")].drop(columns=["_merge", "workbook"])
    retained["source"] = "previous_exact_ticker_quote"
    new = new.copy()
    new["source"] = "2026-10-09_refresh"
    combined = pd.concat([new, retained], ignore_index=True).sort_values(["variable", "date"])
    assert not combined.duplicated(["date", "variable"]).any()
    return combined, retained


def main() -> dict:
    if not REFRESH.exists():
        REFRESH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DOWNLOAD, REFRESH)
    expected = pd.read_csv(INPUTS)
    new, profiles = read_refresh(REFRESH, expected)
    old = exact_history(expected)
    keys = ["date", "variable", "ticker"]
    matched = new.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    scale = expected.set_index("variable").unit.map({"percent": 100., "bps": 1.})
    matched["abs_revision_bps"] = (matched.value_new - matched.value_old).abs() * matched.variable.map(scale)
    reconciliation = matched.groupby("variable").agg(
        n_overlap=("date", "size"), mae_bps=("abs_revision_bps", "mean"),
        max_abs_revision_bps=("abs_revision_bps", "max"))
    profiles = profiles.merge(reconciliation, on="variable", validate="one_to_one")
    combined, retained = retain_observed_history(new, old)
    combined.to_csv(CLEAN / "treasury_refresh_values_long.csv", index=False)
    retained.to_csv(TABLE / "treasury_refresh_retained_history.csv", index=False)
    profiles.to_csv(TABLE / "treasury_refresh_qc.csv", index=False)
    wide = combined.pivot(index="date", columns="variable", values="value").reset_index()
    rebuilt = compute_raw_panel(wide, currencies=COHORT, regimes=["sofr"])
    public = pd.read_csv(CLEAN / "public_cip_cleaned.csv", parse_dates=["date", "cip_govt_break_date"])
    rebuilt = mark_public_break_selection(rebuilt, public)
    rebuilt["input_snapshot"] = "2026-10-09_refresh"
    rebuilt["retained_history_leg_count"] = 0
    for (ccy, tenor), group in rebuilt.groupby(["currency", "tenor"]):
        variables = expected[expected.currency.isin([ccy, "USD"]) & expected.tenor.eq(tenor)].variable
        counts = retained[retained.variable.isin(variables)].groupby("date").size()
        rebuilt.loc[group.index, "retained_history_leg_count"] = group.date.map(counts).fillna(0).astype(int)
    legacy = pd.read_csv(CLEAN / "raw_g10_benchmark_eq6_5y10y_panel.csv", parse_dates=["date"], low_memory=False)
    replacing = legacy.currency.isin(COHORT) & legacy.regime.eq("sofr") & legacy.date.ge("2023-01-01")
    legacy = legacy[~replacing].copy()
    legacy["input_snapshot"] = "six_workbook_baseline"
    legacy["retained_history_leg_count"] = 0
    panel = pd.concat([legacy, rebuilt], ignore_index=True).sort_values(["currency", "tenor", "regime", "date"])
    assert not panel.duplicated(["currency", "tenor", "regime", "date"]).any()
    assert rebuilt.eq6_residual_bps.abs().max() < 1e-8
    panel.to_csv(PANEL, index=False)
    validation = build_validation(panel, public)
    validation.to_csv(TABLE / "treasury_final_replication_validation.csv", index=False)
    govt = validation[validation.check_type.eq("x_govt_vs_public_combined")]
    rho = validation[validation.check_type.eq("rho_vs_public_combined")]
    assert govt.status.eq("GOOD_MATCH").all(), "Refreshed government-CIP validation failed"
    assert rho.status.eq("DIAGNOSTIC_GOOD").all(), "Refreshed rho diagnostic failed"
    receipt = {"source": str(REFRESH.relative_to(ROOT)), "sha256": hashlib.sha256(REFRESH.read_bytes()).hexdigest(),
               "refresh_blocks": len(profiles), "numeric_quotes": len(new),
               "retained_history_quotes": len(retained), "retained_variables": sorted(retained.variable.unique()),
               "min_series_end": str(profiles.last_date.min().date()),
               "max_series_end": str(profiles.last_date.max().date()),
               "max_history_revision_bps": float(matched.abs_revision_bps.max()),
               "government_cip_good_matches": int(govt.status.eq("GOOD_MATCH").sum()),
               "rho_diagnostic_good": int(rho.status.eq("DIAGNOSTIC_GOOD").sum()),
               "eq6_max_residual_bps": float(panel.eq6_residual_bps.abs().max()),
               "notes": ["Historical fallback is an observed same-ticker quote, not forward fill.",
                         "Only the fixed-eight SOFR-regime series are extended; JPY/NOK and other maturities remain baseline.",
                         "The public benchmark ends in June 2025; later dates have no independent public match."]}
    (TABLE / "treasury_refresh_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
