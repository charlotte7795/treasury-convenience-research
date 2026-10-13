from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "output" / "report"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned"

DOWNLOADS = PROJECT_ROOT / "data" / "bloomberg"
if not DOWNLOADS.exists():
    DOWNLOADS = Path.home() / "Downloads"
FILES = {
    "g10_5y": DOWNLOADS / "G10_5Y_Bloomberg_BDH_Clean.xlsx",
    "figure3_g10_term": DOWNLOADS / "Figure3_G10_TermStructure_Bloomberg_BDH_Clean.xlsx",
    "em_5y": DOWNLOADS / "EM_5Y_Government_CIP_Bloomberg_BDH_Clean.xlsx",
    "g10_benchmark_eq6": DOWNLOADS / "G10_Benchmark_CIP_and_Equation6_Bloomberg_BDH_Clean.xlsx",
    "macro_controls": DOWNLOADS / "Macro_and_Financial_Controls_Bloomberg_BDH_Candidate.xlsx",
    "treasury_liquidity_dealer": DOWNLOADS / "Treasury_Liquidity_Dealer_Capacity_Bloomberg_BDH_Candidate.xlsx",
}

UTILITY_SHEETS = {
    "00_README",
    "00_Control",
    "01_Data_Dictionary",
    "02_Access_Log",
    "03_Formula_Index",
    "04_Non_BDH_Data_Needed",
}


def is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_dictionary(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name="01_Data_Dictionary")
    except Exception:
        return pd.DataFrame()
    return df


def parse_workbook_values(label: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wb = load_workbook(path, data_only=True, read_only=True)
    dictionary = load_dictionary(path)
    rows = []
    coverage_rows = []

    for sheet in wb.sheetnames:
        if sheet in UTILITY_SHEETS:
            continue
        ws = wb[sheet]
        sheet_rows = list(ws.iter_rows(values_only=True))
        if len(sheet_rows) < 5:
            continue
        headers = sheet_rows[:4]
        data_rows = sheet_rows[4:]
        max_cols = max(len(r) for r in headers)
        for date_idx in range(0, max_cols, 3):
            value_idx = date_idx + 1
            ticker = headers[0][value_idx] if value_idx < len(headers[0]) else None
            variable = headers[1][value_idx] if value_idx < len(headers[1]) else None
            field_or_unit = headers[2][value_idx] if value_idx < len(headers[2]) else None
            if not ticker or not variable:
                continue
            total = 0
            n_numeric = 0
            n_error = 0
            n_blank = 0
            first_valid = None
            last_valid = None
            latest_value = None
            latest_date = None
            errors = {}
            values = []
            for row in data_rows:
                dt = row[date_idx] if date_idx < len(row) else None
                val = row[value_idx] if value_idx < len(row) else None
                if dt is None and val is None:
                    continue
                total += 1
                dt = pd.to_datetime(dt, errors="coerce")
                if pd.isna(dt):
                    if isinstance(val, str) and val.startswith("#"):
                        errors[val] = errors.get(val, 0) + 1
                        n_error += 1
                    continue
                if is_numeric(val):
                    fval = float(val)
                    n_numeric += 1
                    first_valid = dt if first_valid is None else first_valid
                    last_valid = dt
                    latest_date = dt
                    latest_value = fval
                    values.append(fval)
                    rows.append(
                        {
                            "workbook": label,
                            "sheet": sheet,
                            "date": dt,
                            "ticker": str(ticker),
                            "variable": str(variable),
                            "field_or_unit": str(field_or_unit),
                            "value": fval,
                        }
                    )
                elif isinstance(val, str) and val.startswith("#"):
                    errors[val] = errors.get(val, 0) + 1
                    n_error += 1
                elif val in ("", None):
                    n_blank += 1
                else:
                    n_error += 1
                    key = str(val)
                    errors[key] = errors.get(key, 0) + 1
            s = pd.Series(values, dtype="float64")
            status = "OK"
            if n_numeric == 0:
                status = "NO_NUMERIC_DATA"
            elif n_numeric < 20:
                status = "VERY_FEW_OBS"
            coverage_rows.append(
                {
                    "workbook": label,
                    "sheet": sheet,
                    "ticker": str(ticker),
                    "variable": str(variable),
                    "field_or_unit": str(field_or_unit),
                    "status": status,
                    "first_valid_date": first_valid,
                    "last_valid_date": last_valid,
                    "latest_value": latest_value,
                    "latest_value_date": latest_date,
                    "n_numeric": n_numeric,
                    "n_error_or_text": n_error,
                    "n_blank": n_blank,
                    "n_total_rows": total,
                    "pct_numeric": n_numeric / total if total else 0.0,
                    "median": s.median() if not s.empty else pd.NA,
                    "min": s.min() if not s.empty else pd.NA,
                    "max": s.max() if not s.empty else pd.NA,
                    "error_summary": "; ".join(f"{k}:{v}" for k, v in sorted(errors.items())[:5]),
                    "file_path": str(path),
                }
            )

    values_df = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    if not dictionary.empty and not coverage.empty:
        if "sheet" not in dictionary.columns and "country_sheet" in dictionary.columns:
            dictionary["sheet"] = dictionary["country_sheet"]
        if "field" not in dictionary.columns:
            dictionary["field"] = "PX_LAST"
        dict_cols = [c for c in ["sheet", "ticker", "field", "category", "regime", "type", "tenor", "unit_guess", "name", "floating_leg", "source", "confidence", "note"] if c in dictionary.columns]
        if {"sheet", "ticker"}.issubset(dictionary.columns):
            coverage = coverage.merge(dictionary[dict_cols].drop_duplicates(), on=["sheet", "ticker"], how="left")
    return values_df, coverage


def formula_scan(label: str, path: Path) -> pd.DataFrame:
    wb = load_workbook(path, data_only=False, read_only=True)
    rows = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                val = cell.value
                if isinstance(val, str) and val.startswith("=BDH("):
                    rows.append(
                        {
                            "workbook": label,
                            "sheet": ws.title,
                            "cell": cell.coordinate,
                            "has_days_a": "Days=A" in val,
                            "has_fill_p": "Fill=P" in val,
                            "formula": val,
                        }
                    )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, coverage: pd.DataFrame, formulas: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "Downloaded_Bloomberg_Workbooks_QC.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "00_Summary"
    rows = [
        ("Workbooks checked", len(summary)),
        ("Total ticker blocks", int(summary["ticker_blocks"].sum()) if not summary.empty else 0),
        ("Ticker blocks with no numeric data", int(summary["no_numeric_blocks"].sum()) if not summary.empty else 0),
        ("Executable BDH formulas", int(summary["bdh_formulas"].sum()) if not summary.empty else 0),
        ("BDH formulas with Days=A", int(summary["formulas_with_days_a"].sum()) if not summary.empty else 0),
        ("BDH formulas with Fill=P", int(summary["formulas_with_fill_p"].sum()) if not summary.empty else 0),
    ]
    ws["A1"] = "Downloaded Bloomberg Workbooks QC"
    ws["A1"].font = Font(bold=True, size=16, color="1F4E78")
    for r, (k, v) in enumerate(rows, start=3):
        ws.cell(r, 1, k)
        ws.cell(r, 2, v)
        ws.cell(r, 1).font = Font(bold=True)
        ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 24
    add_df_sheet(wb, "01_Workbook_Summary", summary)
    add_df_sheet(wb, "02_Ticker_Coverage", coverage)
    add_df_sheet(wb, "03_BDH_Formulas", formulas)
    wb.save(out)
    return out


def style_header(cell) -> None:
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="1F4E78")
    cell.alignment = Alignment(wrap_text=True, vertical="top")


def add_df_sheet(wb: Workbook, title: str, df: pd.DataFrame) -> None:
    ws = wb.create_sheet(title)
    if df.empty:
        ws["A1"] = "No rows"
        return
    for c, col in enumerate(df.columns, start=1):
        ws.cell(1, c, col)
        style_header(ws.cell(1, c))
    for r, row in enumerate(df.itertuples(index=False), start=2):
        for c, val in enumerate(row, start=1):
            if pd.isna(val):
                val = None
            ws.cell(r, c, val)
            ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for c, col in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(c)].width = min(max(len(str(col)) + 4, 12), 42)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    all_values = []
    all_coverage = []
    all_formulas = []
    summary_rows = []
    for label, path in FILES.items():
        values, coverage = parse_workbook_values(label, path)
        formulas = formula_scan(label, path)
        all_values.append(values)
        all_coverage.append(coverage)
        all_formulas.append(formulas)
        summary_rows.append(
            {
                "workbook": label,
                "file_path": str(path),
                "file_size_mb": path.stat().st_size / 1_000_000,
                "data_sheets": int(coverage["sheet"].nunique()) if not coverage.empty else 0,
                "ticker_blocks": int(len(coverage)),
                "numeric_observations": int(len(values)),
                "no_numeric_blocks": int(coverage["n_numeric"].eq(0).sum()) if not coverage.empty else 0,
                "short_blocks_lt_20_obs": int(coverage["n_numeric"].lt(20).sum()) if not coverage.empty else 0,
                "bdh_formulas": int(len(formulas)),
                "formulas_with_days_a": int(formulas["has_days_a"].sum()) if not formulas.empty else 0,
                "formulas_with_fill_p": int(formulas["has_fill_p"].sum()) if not formulas.empty else 0,
            }
        )

    values = pd.concat(all_values, ignore_index=True) if all_values else pd.DataFrame()
    coverage = pd.concat(all_coverage, ignore_index=True) if all_coverage else pd.DataFrame()
    formulas = pd.concat(all_formulas, ignore_index=True) if all_formulas else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    values.to_csv(CLEANED_DIR / "downloaded_bloomberg_values_long.csv", index=False)
    coverage.to_csv(TABLE_DIR / "downloaded_bloomberg_ticker_coverage.csv", index=False)
    formulas.to_csv(TABLE_DIR / "downloaded_bloomberg_formula_scan.csv", index=False)
    summary.to_csv(TABLE_DIR / "downloaded_bloomberg_workbook_summary.csv", index=False)
    report = write_report(summary, coverage, formulas)
    print(summary.to_string(index=False))
    print(f"QC report: {report}")


if __name__ == "__main__":
    main()
