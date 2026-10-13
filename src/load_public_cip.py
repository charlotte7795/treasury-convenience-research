"""Load and standardize the public processed CIP dataset."""

from __future__ import annotations

import pandas as pd

from load_and_clean import clean_dataset, inspect_dataset, load_raw_dataset, write_inspection
from utils import CLEANED_DIR, TABLE_DIR, find_public_cip_path, save_csv


def load_public_cleaned() -> pd.DataFrame:
    path = CLEANED_DIR / "public_cip_cleaned.csv"
    if path.exists():
        df = pd.read_csv(path)
        for col in ["date", "cip_govt_break_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    raw_path = find_public_cip_path()
    raw = load_raw_dataset(raw_path)
    cleaned = clean_dataset(raw)
    save_csv(cleaned, path)
    return cleaned


def main() -> None:
    raw_path = find_public_cip_path()
    raw = load_raw_dataset(raw_path)
    info = inspect_dataset(raw, raw_path)
    write_inspection(info)
    cleaned = clean_dataset(raw)
    out = save_csv(cleaned, CLEANED_DIR / "public_cip_cleaned.csv")

    columns = pd.DataFrame(
        [{"standard_name": k, "actual_column": v} for k, v in info["resolved_columns"].items()]
    )
    save_csv(columns, TABLE_DIR / "public_column_mapping.csv")

    print(f"Public CIP shape: {info['shape']}")
    print(f"Date range: {info['date_range']}")
    print(f"Groups: {info['unique_groups']}")
    print(f"Tenors: {info['unique_tenors']}")
    print(f"Currencies: {info['unique_currencies']}")
    print(f"Saved cleaned public CIP data to {out}")


if __name__ == "__main__":
    main()
