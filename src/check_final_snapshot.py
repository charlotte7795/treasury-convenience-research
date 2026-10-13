"""Independently check the final study against same-date raw legs and saved deliverables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COHORT = ["AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "NZD", "SEK"]


def main() -> dict:
    summary = json.loads((ROOT / "output/tables/treasury_final_summary.json").read_text())
    raw = pd.read_csv(ROOT / "output/cleaned/raw_g10_benchmark_eq6_5y10y_refreshed_panel.csv",
                      parse_dates=["date"], low_memory=False)
    selected = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT)
                   & raw.date.between("2023-01-01", "2026-09-30")].copy()
    assert not selected.duplicated(["date", "currency", "tenor"]).any()
    assert selected.date.dt.weekday.lt(5).all()
    selected["month"] = selected.date.dt.to_period("M").dt.to_timestamp("M")
    # Recalculate from the saved instrument legs rather than use the analysis helpers.
    selected["gov_check"] = 100 * (selected.foreign_govt_yield - selected.rho_raw - selected.usd_govt_yield)
    selected["us_spread_check"] = 100 * (selected.usd_swap - selected.usd_govt_yield)
    selected["foreign_spread_check"] = 100 * (selected.foreign_swap - selected.foreign_govt_yield)
    selected["dollar_proxy_check"] = 100 * (selected.foreign_swap - selected.rho_raw - selected.usd_swap)
    np.testing.assert_allclose(selected.gov_check, selected.x_govt_raw_bps, atol=1e-8)
    np.testing.assert_allclose(selected.gov_check,
                               selected.dollar_proxy_check + selected.us_spread_check - selected.foreign_spread_check,
                               atol=1e-8)
    grouped = selected.groupby(["currency", "tenor", "month"])
    counts = grouped.size()
    assert len(counts) == 720 and counts.ge(15).all()
    assert counts.groupby(level=["currency", "tenor"]).size().eq(45).all()
    assert grouped.regime.nunique().eq(1).all()
    months = grouped[["gov_check", "us_spread_check", "foreign_spread_check", "dollar_proxy_check"]].mean().reset_index()
    before = months[months.month.dt.year.eq(2023)].groupby(["currency", "tenor"]).mean(numeric_only=True)
    after = months[months.month.between("2026-01-31", "2026-09-30")].groupby(["currency", "tenor"]).mean(numeric_only=True)
    changes = (after - before).groupby("tenor").mean()
    for row in summary["rebound"]:
        np.testing.assert_allclose(changes.loc[row["tenor"], "gov_check"], row["x_govt_raw_bps"], atol=1e-8)
        np.testing.assert_allclose(changes.loc[row["tenor"], "foreign_spread_check"], row["foreign_swap_spread_bps"], atol=1e-8)
    public = pd.read_csv(ROOT / "output/cleaned/public_cip_cleaned.csv",
                         usecols=["date", "currency", "tenor", "cip_govt"], parse_dates=["date"])
    public["tenor"] = public.tenor.str.lower()
    paired = raw[raw.is_selected_regime_by_public_break].merge(
        public, on=["date", "currency", "tenor"], validate="one_to_one", suffixes=("", "_public"))
    checks = []
    for (ccy, tenor), group in paired.groupby(["currency", "tenor"]):
        group = group[["x_govt_raw_bps", "cip_govt"]].dropna()
        checks.append({"currency": ccy, "tenor": tenor, "n": len(group),
                       "corr": group.x_govt_raw_bps.corr(group.cip_govt),
                       "mae_bps": (group.x_govt_raw_bps - group.cip_govt).abs().mean()})
    assert len(checks) == 20
    assert all(row["corr"] > .98 and row["mae_bps"] < 5 for row in checks)
    executed = ROOT / "notebooks/Decoupling_Dollar_Treasury_Privilege_Raw_Replication_executed.ipynb"
    code = None
    if executed.exists():
        notebook = json.loads(executed.read_text())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") is not None for cell in code)
        assert not any(output["output_type"] == "error" for cell in code for output in cell.get("outputs", []))
    receipt = {"reviewed": "2026-10-13", "data_snapshot": summary["updated"],
               "assessment": "Share with caveats: completed bounded replication plus descriptive extension",
               "country_tenor_months": len(counts), "months_per_pair": 45,
               "minimum_daily_quotes": int(counts.min()), "independent_combined_matches": len(checks),
               "max_mae_bps": max(row["mae_bps"] for row in checks),
               "min_corr": min(row["corr"] for row in checks),
               "rebound_5y_bps": float(changes.loc["5y", "gov_check"]),
               "rebound_10y_bps": float(changes.loc["10y", "gov_check"]),
               "executed_notebook_code_cells": len(code) if code is not None else None,
               "notebook_errors": 0 if code is not None else None}
    out = ROOT / "output/tables/research_readiness_check.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
