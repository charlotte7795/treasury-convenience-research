"""Shared helpers for the raw Bloomberg pilot and public CIP replication."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BBG_DIR = PROJECT_ROOT / "bbg"
OUTPUT_DIR = PROJECT_ROOT / "output"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for directory in [CLEANED_DIR, FIGURE_DIR, TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

TENOR_ORDER = ["3m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]

TICKER_VARIABLE_MAP = {
    "EUXOQQ5": "eur_xccy_5y",
    "EUXOQQ10": "eur_xccy_10y",
    "EESWE5": "eur_swap_5y",
    "EESWE10": "eur_swap_10y",
    "BV050910": "germany_govt_5y",
    "BV100910": "germany_govt_10y",
    "BPXOQQ5": "gbp_xccy_5y",
    "BPXOQQ10": "gbp_xccy_10y",
    "BPSWS5": "gbp_swap_5y",
    "BPSWS10": "gbp_swap_10y",
    "BV050110": "uk_govt_5y",
    "BV100110": "uk_govt_10y",
    "USOSFR5": "usd_swap_5y",
    "USOSFR10": "usd_swap_10y",
    "BV050082": "ust_govt_5y",
    "BV100082": "ust_govt_10y",
    "EUR": "eur_spot",
    "EUR3M": "eur_3m_forward_or_points",
    "GBP": "gbp_spot",
    "GBP3M": "gbp_3m_forward_or_points",
    "JYBSS5Y": "jpy_xccy_5y",
    "JYBSS10Y": "jpy_xccy_10y",
    "JYSO5": "jpy_swap_5y",
    "JYSO10": "jpy_swap_10y",
    "BVCSEI05": "japan_govt_5y",
    "BVCSEI10": "japan_govt_10y",
}


def normalize_ticker(raw: object) -> str | None:
    """Extract a Bloomberg root ticker from messy labels or filenames."""

    if raw is None or pd.isna(raw):
        return None
    text = str(raw).upper().strip()
    text = text.replace(".XLSX", "").replace(".XLS", "")
    text = re.sub(r"\s+", " ", text)
    tokens = re.findall(r"[A-Z0-9]+", text)
    for token in tokens:
        if token in TICKER_VARIABLE_MAP:
            return token
    if tokens:
        return tokens[0]
    return None


def variable_from_ticker(ticker: object) -> str | None:
    root = normalize_ticker(ticker)
    if root is None:
        return None
    return TICKER_VARIABLE_MAP.get(root)


def find_public_cip_path() -> Path:
    candidates = [
        DATA_DIR / "cip_dataset_v4.csv",
        PROJECT_ROOT / "cip_dataset_v4.csv",
        PROJECT_ROOT / "CIP data" / "cip_dataset_v4.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find cip_dataset_v4.csv in data/, project root, or CIP data/.")


def bbg_excel_files() -> list[Path]:
    if not BBG_DIR.exists():
        return []
    return sorted(p for p in BBG_DIR.rglob("*") if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"})


def tenor_sort_key(tenor: str) -> int:
    tenor = str(tenor).lower()
    return TENOR_ORDER.index(tenor) if tenor in TENOR_ORDER else len(TENOR_ORDER)


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
