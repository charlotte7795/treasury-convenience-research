from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
COVERAGE = TABLE_DIR / "downloaded_bloomberg_ticker_coverage.csv"


def has_ok(group: pd.DataFrame, regime: str, typ: str) -> bool:
    mask = group["type"].fillna("").str.lower().eq(typ.lower())
    if regime:
        mask &= group["variable"].str.contains(f"_{regime}_", case=False, na=False)
    return bool(group.loc[mask, "n_numeric"].fillna(0).gt(0).any())


def available_types(group: pd.DataFrame, regime: str) -> str:
    mask = group["variable"].str.contains(f"_{regime}_", case=False, na=False)
    types = sorted(group.loc[mask & group["n_numeric"].fillna(0).gt(0), "type"].dropna().unique())
    return ", ".join(types)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cov = pd.read_csv(COVERAGE)
    em = cov[cov["workbook"].eq("em_5y") & ~cov["sheet"].eq("USD")].copy()

    rows = []
    for ccy, group in em.groupby("sheet"):
        row = {"currency": ccy}
        for regime in ["ibor", "sofr"]:
            has_govt = has_ok(group, regime, "Govt")
            has_rate = has_ok(group, regime, "Rate") or has_ok(group, regime, "Rate benchmark") or has_ok(group, regime, "Rate overnight")
            has_irs = has_ok(group, regime, "IRS")
            has_basis = has_ok(group, regime, "BS") or has_ok(group, regime, "CCS") or has_ok(group, regime, "NDS")
            row[f"{regime}_available_types"] = available_types(group, regime)
            row[f"{regime}_has_govt"] = has_govt
            row[f"{regime}_has_rate_or_irs"] = has_rate or has_irs
            row[f"{regime}_has_basis_ccs_nds"] = has_basis
            row[f"{regime}_constructability"] = "usable_candidate" if has_govt and has_basis and (has_rate or has_irs) else "needs_review"
        no_numeric = group[group["n_numeric"].fillna(0).eq(0)]["ticker"].dropna().tolist()
        row["no_numeric_tickers"] = "; ".join(no_numeric)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("currency")
    out.to_csv(TABLE_DIR / "em_5y_constructability.csv", index=False)
    missing = em[em["n_numeric"].fillna(0).eq(0)][
        ["sheet", "ticker", "variable", "type", "status", "error_summary"]
    ].rename(columns={"sheet": "currency"})
    missing.to_csv(TABLE_DIR / "em_5y_missing_blocks.csv", index=False)
    print(out.to_string(index=False))
    print("Constructability:", TABLE_DIR / "em_5y_constructability.csv")
    print("Missing blocks:", TABLE_DIR / "em_5y_missing_blocks.csv")


if __name__ == "__main__":
    main()
