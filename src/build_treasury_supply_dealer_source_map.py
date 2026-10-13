from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = ROOT / "data" / "source_discovery"
TABLES = ROOT / "output" / "tables"
REPORT = ROOT / "output" / "report"

FISCAL_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
NYFED_BASE = "https://markets.newyorkfed.org/api/pd/get"


def fiscal_api_metadata() -> dict[str, dict[str, str]]:
    path = DISCOVERY / "fiscaldata_mspd_page_data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    apis = data["result"]["pageContext"]["config"]["apis"]
    return {api["endpoint"]: api for api in apis}


def nyfed_metadata() -> dict[str, dict[str, str]]:
    path = DISCOVERY / "nyfed_pd_timeseries_list.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["keyid"]: row for row in data["pd"]["timeseries"]}


def fiscal_row(
    *,
    variable: str,
    endpoint: str,
    fields: str,
    use: str,
    notes: str,
    priority: int,
) -> dict[str, str | int]:
    meta = fiscal_api_metadata()[endpoint]
    return {
        "priority": priority,
        "channel": "Treasury FiscalData",
        "variable": variable,
        "frequency": meta.get("updateFrequency", "Monthly"),
        "coverage": f"{meta.get('earliestDate')} to {meta.get('latestDate')}",
        "official_id_or_endpoint": endpoint,
        "api_url": f"{FISCAL_BASE}/{endpoint}?format=json",
        "fields_to_pull": fields,
        "meaning": meta.get("tableName", ""),
        "research_use": use,
        "bloomberg_needed": "No",
        "bloomberg_search_hint": "Optional backup only: search MSPD / Treasury outstanding / marketable debt.",
        "permission_needed": "No extra permission for official API.",
        "notes": notes,
    }


def nyfed_row(
    *,
    variable: str,
    keyid: str,
    use: str,
    notes: str,
    priority: int,
) -> dict[str, str | int]:
    meta = nyfed_metadata()[keyid]
    return {
        "priority": priority,
        "channel": "NY Fed Primary Dealer Statistics",
        "variable": variable,
        "frequency": "Weekly",
        "coverage": "1998-present page coverage; current SBN2024 key tested from 2013-present",
        "official_id_or_endpoint": keyid,
        "api_url": f"{NYFED_BASE}/{keyid}.json",
        "fields_to_pull": "asofdate, value",
        "meaning": meta.get("description", ""),
        "research_use": use,
        "bloomberg_needed": "No",
        "bloomberg_search_hint": "Optional backup only: search Primary Dealer Statistics / dealer positions.",
        "permission_needed": "No extra permission for official API.",
        "notes": notes,
    }


def build_rows() -> list[dict[str, str | int]]:
    rows = [
        fiscal_row(
            variable="marketable_treasury_outstanding_by_class",
            endpoint="v1/debt/mspd/mspd_table_1",
            fields="record_date, security_type_desc, security_class_desc, debt_held_public_mil_amt, total_mil_amt",
            use="Main Treasury supply control by Bills / Notes / Bonds / TIPS / FRNs.",
            notes="Monthly, clean first-pass supply series. Best immediate replacement for broad FRED debt-to-GDP.",
            priority=1,
        ),
        fiscal_row(
            variable="marketable_treasury_cusip_detail",
            endpoint="v1/debt/mspd/mspd_table_3_market",
            fields="record_date, security_class1_desc, security_class2_desc, issue_date, maturity_date, issued_amt, redeemed_amt, outstanding_amt",
            use="Build maturity-bucket supply: <=2y, 2-5y, 5-10y, 10y+; match Eq.5 tenors.",
            notes="This is the key source for tenor-specific Treasury supply. Needs maturity bucketing after download.",
            priority=1,
        ),
        fiscal_row(
            variable="debt_held_by_public_monthly_total",
            endpoint="v1/debt/mspd/mspd_table_4",
            fields="record_date, security_class_desc, curr_mth_mil_amt",
            use="Broad public-debt control and robustness check.",
            notes="Less precise than marketable/CUSIP supply, but useful for levels and sanity checks.",
            priority=2,
        ),
        {
            "priority": 2,
            "channel": "Treasury FiscalData",
            "variable": "debt_held_by_public_daily_total",
            "frequency": "Daily",
            "coverage": "1993-present from Debt to the Penny API",
            "official_id_or_endpoint": "v2/accounting/od/debt_to_penny",
            "api_url": f"{FISCAL_BASE}/v2/accounting/od/debt_to_penny?format=json",
            "fields_to_pull": "record_date, debt_held_public_amt, intragov_hold_amt, tot_pub_debt_out_amt",
            "meaning": "Daily total public debt outstanding.",
            "research_use": "High-frequency broad supply control; not maturity-specific.",
            "bloomberg_needed": "No",
            "bloomberg_search_hint": "Optional backup only.",
            "permission_needed": "No extra permission for official API.",
            "notes": "Daily total is convenient, but it does not isolate marketable Treasury supply or maturity buckets.",
        },
        nyfed_row(
            variable="dealer_total_ust_position_ex_tips",
            keyid="PDPOSGST-TOT",
            use="Main dealer intermediation/capacity proxy: net Treasury position excluding TIPS.",
            notes="Interpret as long minus short. Confirm units before reporting; API values are typically in millions.",
            priority=1,
        ),
        nyfed_row(
            variable="dealer_bills_position_ex_tips",
            keyid="PDPOSGS-B",
            use="Short-end dealer position proxy.",
            notes="Useful because Bills can behave differently from coupons around debt-ceiling/T-bill supply episodes.",
            priority=2,
        ),
        nyfed_row(
            variable="dealer_coupon_position_le_2y",
            keyid="PDPOSGSC-L2",
            use="Dealer position bucket for front-end coupons.",
            notes="Use with maturity-specific convenience/supply if expanding beyond 5Y/10Y.",
            priority=2,
        ),
        nyfed_row(
            variable="dealer_coupon_position_2y_to_3y",
            keyid="PDPOSGSC-G2L3",
            use="Dealer position bucket for 2-3Y coupons.",
            notes="Tenor bucket for maturity-structure extension.",
            priority=3,
        ),
        nyfed_row(
            variable="dealer_coupon_position_3y_to_6y",
            keyid="PDPOSGSC-G3L6",
            use="Closest dealer-position bucket for 5Y Treasury convenience.",
            notes="Important if testing whether 5Y convenience is related to same-maturity dealer balance-sheet use.",
            priority=1,
        ),
        nyfed_row(
            variable="dealer_coupon_position_6y_to_7y",
            keyid="PDPOSGSC-G6L7",
            use="Intermediate coupon dealer-position bucket.",
            notes="Transition bucket between 5Y and 10Y.",
            priority=3,
        ),
        nyfed_row(
            variable="dealer_coupon_position_7y_to_11y",
            keyid="PDPOSGSC-G7L11",
            use="Closest dealer-position bucket for 10Y Treasury convenience.",
            notes="Important if testing whether 10Y convenience is related to same-maturity dealer balance-sheet use.",
            priority=1,
        ),
        nyfed_row(
            variable="dealer_coupon_position_11y_to_21y",
            keyid="PDPOSGSC-G11L21",
            use="Long-end dealer-position bucket.",
            notes="Useful if later extending Figure 3 / term-structure regressions.",
            priority=3,
        ),
        nyfed_row(
            variable="dealer_coupon_position_gt_21y",
            keyid="PDPOSGSC-G21",
            use="Ultra-long-end dealer-position bucket.",
            notes="Useful for 30Y extension, not first pass.",
            priority=3,
        ),
        nyfed_row(
            variable="dealer_ust_transactions_ex_tips",
            keyid="PDGSWOEXTTOT",
            use="Dealer Treasury trading activity / market-making flow proxy.",
            notes="Can help distinguish balance-sheet position from transaction volume.",
            priority=2,
        ),
        nyfed_row(
            variable="dealer_repo_treasury_ex_tips",
            keyid="PDSORA-UTSETTOT",
            use="Treasury repo funding / balance-sheet usage proxy.",
            notes="NY Fed reports suppressed values as '*'; convert those to missing.",
            priority=1,
        ),
        nyfed_row(
            variable="dealer_reverse_repo_treasury_ex_tips",
            keyid="PDSIRRA-UTSETTOT",
            use="Treasury reverse-repo financing / collateral demand proxy.",
            notes="NY Fed reports suppressed values as '*'; convert those to missing.",
            priority=1,
        ),
        nyfed_row(
            variable="dealer_securities_borrowed_treasury_ex_tips",
            keyid="PDSIOSB-UTSETTOT",
            use="Other dealer financing / securities-borrowing proxy.",
            notes="Secondary balance-sheet proxy; use after repo and position series.",
            priority=3,
        ),
        nyfed_row(
            variable="dealer_securities_lent_treasury_ex_tips",
            keyid="PDSOOS-UTSETTOT",
            use="Other dealer financing / securities-lending proxy.",
            notes="Secondary balance-sheet proxy; use after repo and position series.",
            priority=3,
        ),
        nyfed_row(
            variable="dealer_fails_to_receive_treasury_ex_tips",
            keyid="PDFTR-USTET",
            use="Treasury settlement stress / market-functioning proxy.",
            notes="Useful as a stress/liquidity control rather than a pure capacity variable.",
            priority=2,
        ),
    ]
    return rows


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    priority = df[df["priority"].astype(int) <= 2].copy()
    lines = [
        "# Treasury Supply and Dealer-Capacity Data Source Map",
        "",
        "This file maps the next Eq.5 extension data sources to official public APIs.",
        "Bloomberg is not required for the core supply/dealer proxies below; Bloomberg can be used only as backup.",
        "",
        "## Immediate Pull List",
        "",
    ]
    for _, row in priority.iterrows():
        lines.extend(
            [
                f"### P{row['priority']} - {row['variable']}",
                f"- Source: {row['channel']}",
                f"- ID / endpoint: `{row['official_id_or_endpoint']}`",
                f"- Frequency / coverage: {row['frequency']}; {row['coverage']}",
                f"- Meaning: {row['meaning']}",
                f"- Use: {row['research_use']}",
                f"- API: {row['api_url']}",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_excel(df: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="source_map")
        df[df["priority"].astype(int) == 1].to_excel(writer, index=False, sheet_name="priority_1")
        df[df["channel"].str.contains("FiscalData")].to_excel(writer, index=False, sheet_name="treasury_supply")
        df[df["channel"].str.contains("NY Fed")].to_excel(writer, index=False, sheet_name="dealer_capacity")
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                letter = col[0].column_letter
                max_len = min(max(len(str(cell.value or "")) for cell in col) + 2, 70)
                ws.column_dimensions[letter].width = max_len


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(build_rows()).sort_values(["priority", "channel", "variable"])
    csv_path = TABLES / "treasury_supply_dealer_source_map.csv"
    xlsx_path = REPORT / "Treasury_Supply_Dealer_Data_Source_Map.xlsx"
    md_path = REPORT / "Treasury_Supply_Dealer_Data_Source_Map.md"

    df.to_csv(csv_path, index=False)
    write_excel(df, xlsx_path)
    write_markdown(df, md_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {xlsx_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
