"""Load, inspect, and clean the public Schreger CIP dataset.

This is a processed-data replication helper. It does not reconstruct any
Bloomberg inputs.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"

TENOR_ORDER = ["3m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]
NUMERIC_COLUMNS = [
    "diff_y",
    "rho",
    "cip_govt",
    "rho_ibor",
    "rho_sofr",
    "cip_govt_ibor",
    "cip_govt_sofr",
]

EXPECTED_COLUMNS = {
    "date": ["date", "dt", "time"],
    "currency": ["currency", "curr", "ccy", "country"],
    "group": ["group", "market_group", "sample_group"],
    "tenor": ["tenor", "maturity", "mat"],
    "rho": ["rho", "forward_premium"],
    "diff_y": ["diff_y", "yield_diff", "dy"],
    "cip_govt": ["cip_govt", "govt_cip", "x_govt"],
    "cip_govt_break_date": ["cip_govt_break_date", "break_date"],
    "rho_ibor": ["rho_ibor"],
    "rho_sofr": ["rho_sofr"],
    "cip_govt_ibor": ["cip_govt_ibor"],
    "cip_govt_sofr": ["cip_govt_sofr"],
}


def candidate_data_paths() -> list[Path]:
    return [
        PROJECT_ROOT / "data" / "cip_dataset_v4.csv",
        PROJECT_ROOT / "cip_dataset_v4.csv",
        PROJECT_ROOT / "CIP data" / "cip_dataset_v4.csv",
        PROJECT_ROOT / "Drive uploads" / "cip_dataset_v4_g10_5y_10y_subset.csv",
    ]


def find_data_path() -> Path:
    for path in candidate_data_paths():
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidate_data_paths())
    raise FileNotFoundError(f"Could not find cip_dataset_v4.csv. Searched:\n{searched}")


def _find_column(columns: Iterable[str], expected_name: str) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    aliases = EXPECTED_COLUMNS.get(expected_name, [expected_name])
    for alias in aliases:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]

    close = difflib.get_close_matches(expected_name.lower(), lower_to_original, n=3, cutoff=0.75)
    if close:
        print(
            f"WARNING: Column '{expected_name}' not found exactly. "
            f"Close matches: {[lower_to_original[c] for c in close]}"
        )
    else:
        print(f"WARNING: Column '{expected_name}' not found and no close match detected.")
    return None


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for standard_name in EXPECTED_COLUMNS:
        found = _find_column(df.columns, standard_name)
        if found is not None:
            mapping[standard_name] = found
    return mapping


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or find_data_path()
    return pd.read_csv(path)


def inspect_dataset(df: pd.DataFrame, path: Path) -> dict[str, object]:
    mapping = resolve_columns(df)
    date_col = mapping.get("date")

    date_range = None
    if date_col:
        parsed = pd.to_datetime(df[date_col], format="%d%b%Y", errors="coerce")
        if parsed.isna().mean() > 0.5:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
        date_range = (parsed.min(), parsed.max())

    info = {
        "data_path": str(path),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "date_range": date_range,
        "unique_currencies": sorted(df[mapping["currency"]].dropna().astype(str).unique().tolist())
        if "currency" in mapping
        else [],
        "unique_groups": sorted(df[mapping["group"]].dropna().astype(str).unique().tolist())
        if "group" in mapping
        else [],
        "unique_tenors": sorted(df[mapping["tenor"]].dropna().astype(str).unique().tolist())
        if "tenor" in mapping
        else [],
        "resolved_columns": mapping,
    }
    return info


def write_inspection(info: dict[str, object]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("Dataset inspection")
    lines.append("==================")
    lines.append(f"Data path: {info['data_path']}")
    lines.append(f"Shape: {info['shape']}")
    lines.append("")
    lines.append("Columns:")
    lines.extend(f"- {c}" for c in info["columns"])
    lines.append("")
    lines.append("Dtypes:")
    lines.extend(f"- {k}: {v}" for k, v in info["dtypes"].items())
    lines.append("")
    lines.append("Missing values:")
    lines.extend(f"- {k}: {v}" for k, v in info["missing_values"].items())
    lines.append("")
    lines.append(f"Date range: {info['date_range']}")
    lines.append(f"Unique groups: {info['unique_groups']}")
    lines.append(f"Unique tenors: {info['unique_tenors']}")
    lines.append(f"Unique currencies/countries: {info['unique_currencies']}")
    lines.append("")
    lines.append("Resolved columns:")
    lines.extend(f"- {k}: {v}" for k, v in info["resolved_columns"].items())
    (TABLE_DIR / "dataset_inspection.txt").write_text("\n".join(lines), encoding="utf-8")

    missing_df = pd.DataFrame(
        sorted(info["missing_values"].items()), columns=["column", "missing_count"]
    )
    missing_df.to_csv(TABLE_DIR / "missing_value_counts.csv", index=False)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    mapping = resolve_columns(df)
    required = ["date", "currency", "group", "tenor", "cip_govt"]
    missing_required = [c for c in required if c not in mapping]
    if missing_required:
        raise ValueError(f"Missing required columns after resolution: {missing_required}")

    cleaned = df.copy()

    rename_map = {actual: standard for standard, actual in mapping.items() if actual != standard}
    if rename_map:
        cleaned = cleaned.rename(columns=rename_map)

    cleaned["date"] = pd.to_datetime(cleaned["date"], format="%d%b%Y", errors="coerce")
    if cleaned["date"].isna().mean() > 0.5:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    cleaned["currency"] = cleaned["currency"].astype(str).str.upper().str.strip()
    cleaned["group"] = cleaned["group"].astype(str).str.lower().str.strip()
    cleaned["group_label"] = cleaned["group"].map({"g10": "G10", "eme": "EM"}).fillna(
        cleaned["group"].str.upper()
    )
    cleaned["tenor"] = cleaned["tenor"].astype(str).str.lower().str.strip()
    cleaned["tenor"] = pd.Categorical(cleaned["tenor"], categories=TENOR_ORDER, ordered=True)

    if "cip_govt_break_date" in cleaned.columns:
        cleaned["cip_govt_break_date"] = pd.to_datetime(
            cleaned["cip_govt_break_date"], format="%d%b%Y", errors="coerce"
        )

    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.sort_values(["date", "group", "currency", "tenor"]).reset_index(drop=True)
    return cleaned


def load_cleaned_dataset() -> pd.DataFrame:
    cleaned_path = OUTPUT_DIR / "cleaned_cip_dataset.csv"
    if not cleaned_path.exists():
        raw_path = find_data_path()
        raw = load_raw_dataset(raw_path)
        cleaned = clean_dataset(raw)
        save_cleaned_dataset(cleaned)
        return cleaned

    df = pd.read_csv(cleaned_path)
    for date_col in ["date", "cip_govt_break_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "tenor" in df.columns:
        df["tenor"] = pd.Categorical(df["tenor"], categories=TENOR_ORDER, ordered=True)
    return df


def save_cleaned_dataset(cleaned: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "cleaned_cip_dataset.csv"
    cleaned.to_csv(path, index=False)
    return path


def g10_filter(df: pd.DataFrame) -> pd.Series:
    return df["group"].astype(str).str.lower().eq("g10")


def tenor_filter(df: pd.DataFrame, tenor: str) -> pd.Series:
    return df["tenor"].astype(str).str.lower().eq(tenor.lower())


def main() -> None:
    data_path = find_data_path()
    print(f"Loading {data_path}")
    raw = load_raw_dataset(data_path)
    info = inspect_dataset(raw, data_path)
    write_inspection(info)

    print("Shape:", info["shape"])
    print("Columns:", info["columns"])
    print("Date range:", info["date_range"])
    print("Groups:", info["unique_groups"])
    print("Tenors:", info["unique_tenors"])
    print("Currencies:", info["unique_currencies"])
    print("Resolved columns:", info["resolved_columns"])

    cleaned = clean_dataset(raw)
    cleaned_path = save_cleaned_dataset(cleaned)
    print(f"Saved cleaned dataset to {cleaned_path}")


if __name__ == "__main__":
    main()
