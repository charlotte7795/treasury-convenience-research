from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_DIR = PROJECT_ROOT / "output" / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned"
DATA_DIR = PROJECT_ROOT / "data" / "official"
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
REPORT_DIR = PROJECT_ROOT / "output" / "report"

FISCAL_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
NYFED_BASE = "https://markets.newyorkfed.org/api/pd/get"

FISCAL_TABLES = {
    "mspd_table_1": "v1/debt/mspd/mspd_table_1",
    "mspd_table_3_market": "v1/debt/mspd/mspd_table_3_market",
    "mspd_table_4": "v1/debt/mspd/mspd_table_4",
}

NYFED_SERIES = {
    "dealer_total_ust_position_ex_tips": "PDPOSGST-TOT",
    "dealer_bills_position_ex_tips": "PDPOSGS-B",
    "dealer_coupon_position_le_2y": "PDPOSGSC-L2",
    "dealer_coupon_position_3y_to_6y": "PDPOSGSC-G3L6",
    "dealer_coupon_position_7y_to_11y": "PDPOSGSC-G7L11",
    "dealer_repo_treasury_ex_tips": "PDSORA-UTSETTOT",
    "dealer_reverse_repo_treasury_ex_tips": "PDSIRRA-UTSETTOT",
    "dealer_ust_transactions_ex_tips": "PDGSWOEXTTOT",
    "dealer_fails_to_receive_treasury_ex_tips": "PDFTR-USTET",
}

TARGETS = ["x_govt_raw_bps_5y", "x_govt_raw_bps_10y"]
CONTROL_VARS = [
    "bbdxy_index",
    "vix_index",
    "move_index",
    "us_generic_govt_5y_yield",
    "ust_10y_2y_slope",
]


def get_json(url: str, retries: int = 4) -> dict:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "samim-cip-replication/1.0"})
            with urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - preserve network failure detail for research logs.
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_exc}") from last_exc


def fiscal_url(endpoint: str, page_number: int, page_size: int) -> str:
    query = urlencode(
        {
            "format": "json",
            "page[number]": page_number,
            "page[size]": page_size,
            "sort": "record_date",
        }
    )
    return f"{FISCAL_BASE}/{endpoint}?{query}"


def download_fiscal_table(name: str, endpoint: str, page_size: int = 10_000) -> pd.DataFrame:
    cache = DATA_DIR / f"fiscal_{name}.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)

    first = get_json(fiscal_url(endpoint, 1, page_size))
    rows = list(first.get("data", []))
    total_pages = int(first.get("meta", {}).get("total-pages", 1))
    for page in range(2, total_pages + 1):
        data = get_json(fiscal_url(endpoint, page, page_size))
        rows.extend(data.get("data", []))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def download_nyfed_series(variable: str, keyid: str) -> pd.DataFrame:
    cache = DATA_DIR / f"nyfed_{keyid}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    data = get_json(f"{NYFED_BASE}/{keyid}.json")
    rows = data.get("pd", {}).get("timeseries", [])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["asofdate"], errors="coerce")
    out["variable"] = variable
    out["value"] = pd.to_numeric(out["value"].replace("*", np.nan), errors="coerce")
    out = out[["date", "keyid", "variable", "value"]].dropna(subset=["date"])
    out.to_csv(cache, index=False)
    return out


def clean_mspd_table_1(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["record_date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    data["amount_mil"] = pd.to_numeric(data["debt_held_public_mil_amt"], errors="coerce")
    marketable = data[data["security_type_desc"].eq("Marketable")].copy()
    pivot = marketable.pivot_table(
        index="date",
        columns="security_class_desc",
        values="amount_mil",
        aggfunc="sum",
    )
    rename = {
        "Bills": "marketable_bills_mil",
        "Notes": "marketable_notes_mil",
        "Bonds": "marketable_bonds_mil",
        "Treasury Inflation-Protected Securities": "marketable_tips_mil",
        "Floating Rate Notes": "marketable_frns_mil",
    }
    pivot = pivot.rename(columns=rename)
    out = pivot.reset_index()
    amount_cols = [c for c in out.columns if c.startswith("marketable_") and c.endswith("_mil")]
    out["marketable_total_mil"] = out[amount_cols].sum(axis=1)
    for col in amount_cols:
        out[col.replace("_mil", "_share")] = out[col] / out["marketable_total_mil"]
    return out.sort_values("date")


def clean_mspd_table_3_market(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["record_date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    data["maturity_date"] = pd.to_datetime(data["maturity_date"], errors="coerce")
    data["outstanding_amt"] = pd.to_numeric(data["outstanding_amt"], errors="coerce")
    data = data.dropna(subset=["date", "maturity_date", "outstanding_amt"])
    data["remaining_years"] = (data["maturity_date"] - data["date"]).dt.days / 365.25
    data["outstanding_mil"] = data["outstanding_amt"]

    bins = [-np.inf, 2, 3, 6, 7, 11, 21, np.inf]
    labels = ["le_2y", "2y_to_3y", "3y_to_6y", "6y_to_7y", "7y_to_11y", "11y_to_21y", "gt_21y"]
    data["maturity_bucket"] = pd.cut(data["remaining_years"], bins=bins, labels=labels)
    pivot = data.pivot_table(
        index="date",
        columns="maturity_bucket",
        values="outstanding_mil",
        aggfunc="sum",
        observed=False,
    ).reset_index()
    pivot = pivot.rename(columns={bucket: f"marketable_maturity_{bucket}_mil" for bucket in labels})
    bucket_cols = [c for c in pivot.columns if c.startswith("marketable_maturity_")]
    pivot["marketable_maturity_bucket_total_mil"] = pivot[bucket_cols].sum(axis=1)
    return pivot.sort_values("date")


def clean_mspd_table_4(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["record_date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    data["curr_mth_mil_amt"] = pd.to_numeric(data["curr_mth_mil_amt"], errors="coerce")
    pivot = data.pivot_table(
        index="date",
        columns="security_class_desc",
        values="curr_mth_mil_amt",
        aggfunc="sum",
    ).reset_index()
    return pivot.rename(
        columns={
            "Debt Held by the Public": "debt_held_by_public_mil",
            "Intragovernmental Holdings": "intragov_holdings_mil",
            "Total Public Debt Outstanding": "total_public_debt_mil",
        }
    ).sort_values("date")


def build_supply_monthly() -> pd.DataFrame:
    table1 = download_fiscal_table("mspd_table_1", FISCAL_TABLES["mspd_table_1"])
    table3 = download_fiscal_table("mspd_table_3_market", FISCAL_TABLES["mspd_table_3_market"])
    table4 = download_fiscal_table("mspd_table_4", FISCAL_TABLES["mspd_table_4"])
    supply = clean_mspd_table_1(table1)
    maturity = clean_mspd_table_3_market(table3)
    broad = clean_mspd_table_4(table4)
    out = supply.merge(maturity, on="date", how="outer").merge(broad, on="date", how="outer").sort_values("date")
    out.to_csv(CLEANED_DIR / "treasury_supply_official_monthly.csv", index=False)
    return out


def build_dealer_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    long_frames = [download_nyfed_series(variable, keyid) for variable, keyid in NYFED_SERIES.items()]
    weekly = pd.concat(long_frames, ignore_index=True).sort_values(["variable", "date"])
    weekly.to_csv(CLEANED_DIR / "nyfed_primary_dealer_selected_weekly_long.csv", index=False)

    weekly["date_month"] = weekly["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = weekly.groupby(["date_month", "variable"], as_index=False)["value"].mean()
    wide = monthly.pivot(index="date_month", columns="variable", values="value").reset_index()
    wide = wide.rename(columns={"date_month": "date"}).sort_values("date")
    wide.to_csv(CLEANED_DIR / "nyfed_primary_dealer_selected_monthly.csv", index=False)
    return weekly, wide


def add_changes(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values("date").copy()
    for col in panel.columns:
        if col == "date":
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            panel[f"{col}_change_12m"] = panel[col].diff(12)
            if (panel[col] > 0).all(skipna=True):
                panel[f"{col}_growth_12m_pct"] = panel[col].pct_change(12) * 100.0
    return panel


def build_panel() -> pd.DataFrame:
    base = pd.read_csv(CLEANED_DIR / "eq5_ust_extension_monthly_panel.csv", parse_dates=["date"])
    supply = build_supply_monthly()
    _, dealer = build_dealer_monthly()
    panel = base.merge(supply, on="date", how="left").merge(dealer, on="date", how="left")
    panel = add_changes(panel)
    panel.to_csv(CLEANED_DIR / "eq5_ust_supply_dealer_official_monthly_panel.csv", index=False)
    return panel


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std()
    if not std or np.isnan(std):
        return x * np.nan
    return (x - x.mean()) / std


def variable_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in panel.columns:
        if col == "date" or not pd.api.types.is_numeric_dtype(panel[col]):
            continue
        valid = panel[["date", col]].dropna()
        rows.append(
            {
                "variable": col,
                "n_months": len(valid),
                "date_min": valid["date"].min().date().isoformat() if len(valid) else "",
                "date_max": valid["date"].max().date().isoformat() if len(valid) else "",
                "latest_value": valid[col].iloc[-1] if len(valid) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["n_months", "variable"], ascending=[False, True])
    out.to_csv(TABLE_DIR / "eq5_supply_dealer_official_variable_coverage.csv", index=False)
    return out


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "marketable_total_mil",
        "marketable_bills_share",
        "marketable_notes_share",
        "marketable_maturity_3y_to_6y_mil",
        "marketable_maturity_7y_to_11y_mil",
        "dealer_total_ust_position_ex_tips",
        "dealer_coupon_position_3y_to_6y",
        "dealer_coupon_position_7y_to_11y",
        "dealer_repo_treasury_ex_tips",
        "dealer_reverse_repo_treasury_ex_tips",
        "dealer_fails_to_receive_treasury_ex_tips",
    ]
    rows = []
    for target in TARGETS:
        for var in variables:
            if var not in panel.columns:
                continue
            sub = panel[["date", target, var]].dropna()
            change_cols = [f"{target}_change_12m", f"{var}_change_12m"]
            sub_change = panel[["date", *change_cols]].dropna() if all(c in panel.columns for c in change_cols) else pd.DataFrame()
            rows.append(
                {
                    "target": target,
                    "variable": var,
                    "n_level": len(sub),
                    "level_date_min": sub["date"].min().date().isoformat() if len(sub) else "",
                    "level_date_max": sub["date"].max().date().isoformat() if len(sub) else "",
                    "corr_level": sub[target].corr(sub[var]) if len(sub) >= 24 else np.nan,
                    "n_change_12m": len(sub_change),
                    "corr_change_12m": sub_change[change_cols[0]].corr(sub_change[change_cols[1]]) if len(sub_change) >= 24 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "eq5_supply_dealer_official_correlations.csv", index=False)
    return out


def run_model(panel: pd.DataFrame, target: str, spec: str, y_col: str, x_cols: list[str]) -> tuple[dict, list[dict]]:
    data = panel[["date", y_col, *x_cols]].dropna().copy()
    if len(data) < 48:
        return {}, []
    y = pd.to_numeric(data[y_col], errors="coerce")
    x = data[x_cols].apply(zscore)
    model_data = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(model_data) < 48:
        return {}, []
    model = sm.OLS(model_data["y"], sm.add_constant(model_data.drop(columns=["y"]))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 6},
    )
    model_row = {
        "target": target,
        "spec": spec,
        "n_months": int(model.nobs),
        "date_min": data.loc[model_data.index, "date"].min().date().isoformat(),
        "date_max": data.loc[model_data.index, "date"].max().date().isoformat(),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }
    coef_rows = [
        {
            "target": target,
            "spec": spec,
            "variable": col,
            "coef_bps_per_1sd": model.params.get(col, np.nan),
            "t_stat_hac": model.tvalues.get(col, np.nan),
            "p_value_hac": model.pvalues.get(col, np.nan),
            "n_months": int(model.nobs),
            "r_squared": model.rsquared,
        }
        for col in x_cols
    ]
    return model_row, coef_rows


def regression_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs_by_target = {
        "x_govt_raw_bps_5y": {
            "official_supply_level": ["marketable_total_mil", "marketable_maturity_3y_to_6y_mil"],
            "dealer_capacity_level": [
                "dealer_total_ust_position_ex_tips",
                "dealer_coupon_position_3y_to_6y",
                "dealer_repo_treasury_ex_tips",
                "dealer_reverse_repo_treasury_ex_tips",
            ],
            "supply_dealer_controls_level": [
                "marketable_maturity_3y_to_6y_mil",
                "dealer_coupon_position_3y_to_6y",
                "dealer_repo_treasury_ex_tips",
                "bbdxy_index",
                "vix_index",
                "move_index",
                "us_generic_govt_5y_yield",
                "ust_10y_2y_slope",
            ],
            "supply_dealer_controls_12m_change": [
                "marketable_maturity_3y_to_6y_mil_change_12m",
                "dealer_coupon_position_3y_to_6y_change_12m",
                "dealer_repo_treasury_ex_tips_change_12m",
                "bbdxy_index_change_12m",
                "vix_index_change_12m",
                "move_index_change_12m",
                "us_generic_govt_5y_yield_change_12m",
                "ust_10y_2y_slope_change_12m",
            ],
        },
        "x_govt_raw_bps_10y": {
            "official_supply_level": ["marketable_total_mil", "marketable_maturity_7y_to_11y_mil"],
            "dealer_capacity_level": [
                "dealer_total_ust_position_ex_tips",
                "dealer_coupon_position_7y_to_11y",
                "dealer_repo_treasury_ex_tips",
                "dealer_reverse_repo_treasury_ex_tips",
            ],
            "supply_dealer_controls_level": [
                "marketable_maturity_7y_to_11y_mil",
                "dealer_coupon_position_7y_to_11y",
                "dealer_repo_treasury_ex_tips",
                "bbdxy_index",
                "vix_index",
                "move_index",
                "us_generic_govt_5y_yield",
                "ust_10y_2y_slope",
            ],
            "supply_dealer_controls_12m_change": [
                "marketable_maturity_7y_to_11y_mil_change_12m",
                "dealer_coupon_position_7y_to_11y_change_12m",
                "dealer_repo_treasury_ex_tips_change_12m",
                "bbdxy_index_change_12m",
                "vix_index_change_12m",
                "move_index_change_12m",
                "us_generic_govt_5y_yield_change_12m",
                "ust_10y_2y_slope_change_12m",
            ],
        },
    }
    model_rows = []
    coef_rows = []
    for target, specs in specs_by_target.items():
        for spec, cols in specs.items():
            cols = [col for col in cols if col in panel.columns]
            y_col = f"{target}_change_12m" if spec.endswith("12m_change") else target
            model_row, rows = run_model(panel, target, spec, y_col, cols)
            if model_row:
                model_rows.append(model_row)
                coef_rows.extend(rows)
    models = pd.DataFrame(model_rows)
    coefs = pd.DataFrame(coef_rows)
    models.to_csv(TABLE_DIR / "eq5_supply_dealer_official_regression_models.csv", index=False)
    coefs.to_csv(TABLE_DIR / "eq5_supply_dealer_official_regression_coefficients.csv", index=False)
    return coefs, models


def make_figure(panel: pd.DataFrame) -> Path:
    out = FIGURE_DIR / "eq5_supply_dealer_official_first_pass.png"
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    axes[0].plot(panel["date"], panel["x_govt_raw_bps_5y"], label="G10 median Eq.5 UST convenience 5Y", lw=1.2)
    axes[0].plot(panel["date"], panel["x_govt_raw_bps_10y"], label="G10 median Eq.5 UST convenience 10Y", lw=1.2)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("bps")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Eq.5 UST Convenience With Official Supply / Dealer Proxies")

    axes[1].plot(panel["date"], zscore(panel["marketable_maturity_3y_to_6y_mil"]), label="Marketable supply 3-6Y z", lw=1.0)
    axes[1].plot(panel["date"], zscore(panel["marketable_maturity_7y_to_11y_mil"]), label="Marketable supply 7-11Y z", lw=1.0)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("z-score")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(panel["date"], zscore(panel["dealer_coupon_position_3y_to_6y"]), label="Dealer position 3-6Y z", lw=1.0)
    axes[2].plot(panel["date"], zscore(panel["dealer_coupon_position_7y_to_11y"]), label="Dealer position 7-11Y z", lw=1.0)
    axes[2].plot(panel["date"], zscore(panel["dealer_repo_treasury_ex_tips"]), label="Dealer Treasury repo z", lw=1.0, alpha=0.8)
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_ylabel("z-score")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_memo(panel: pd.DataFrame, coverage: pd.DataFrame, corr: pd.DataFrame, coefs: pd.DataFrame, models: pd.DataFrame, figure: Path) -> Path:
    def coef(target: str, spec: str, variable: str) -> pd.Series | None:
        rows = coefs[(coefs["target"].eq(target)) & (coefs["spec"].eq(spec)) & (coefs["variable"].eq(variable))]
        return None if rows.empty else rows.iloc[0]

    def corr_value(target: str, variable: str, col: str) -> float:
        rows = corr[(corr["target"].eq(target)) & (corr["variable"].eq(variable))]
        return float(rows.iloc[0][col]) if not rows.empty else np.nan

    latest = panel.dropna(subset=["x_govt_raw_bps_5y"]).iloc[-1]
    supply_dates = panel.loc[panel["marketable_total_mil"].notna(), "date"]
    dealer_dates = panel.loc[panel["dealer_total_ust_position_ex_tips"].notna(), "date"]
    c_5_supply = corr_value("x_govt_raw_bps_5y", "marketable_maturity_3y_to_6y_mil", "corr_level")
    c_10_supply = corr_value("x_govt_raw_bps_10y", "marketable_maturity_7y_to_11y_mil", "corr_level")
    c_5_dealer = corr_value("x_govt_raw_bps_5y", "dealer_coupon_position_3y_to_6y", "corr_level")
    c_10_dealer = corr_value("x_govt_raw_bps_10y", "dealer_coupon_position_7y_to_11y", "corr_level")

    beta_5_supply = coef("x_govt_raw_bps_5y", "supply_dealer_controls_level", "marketable_maturity_3y_to_6y_mil")
    beta_5_dealer = coef("x_govt_raw_bps_5y", "supply_dealer_controls_level", "dealer_coupon_position_3y_to_6y")
    beta_10_supply = coef("x_govt_raw_bps_10y", "supply_dealer_controls_level", "marketable_maturity_7y_to_11y_mil")
    beta_10_dealer = coef("x_govt_raw_bps_10y", "supply_dealer_controls_level", "dealer_coupon_position_7y_to_11y")

    lines = [
        "# Eq.5 Official Supply / Dealer Extension First Pass",
        "",
        "Generated: 2026-07-07",
        "",
        "## What Changed",
        "",
        "The Eq.5 direct UST-convenience monthly panel is now merged with official Treasury FiscalData supply variables and NY Fed Primary Dealer Statistics variables.",
        "",
        "This replaces the earlier broad FRED-only supply proxy with more precise public data:",
        "",
        "- marketable Treasury outstanding by security class;",
        "- CUSIP-level marketable Treasury outstanding bucketed by remaining maturity;",
        "- primary dealer Treasury positions by maturity bucket;",
        "- Treasury repo, reverse repo, transactions, and fails proxies.",
        "",
        "## Coverage",
        "",
        f"- Final panel rows: {len(panel)} monthly observations from {panel['date'].min().date().isoformat()} to {panel['date'].max().date().isoformat()}.",
        f"- Treasury FiscalData supply coverage in this panel runs from {supply_dates.min().date().isoformat()} to {supply_dates.max().date().isoformat()}.",
        f"- Dealer series in the current NY Fed SBN2024 API run from {dealer_dates.min().date().isoformat()} to {dealer_dates.max().date().isoformat()} in this panel, so dealer regressions are a post-2013 first pass.",
        f"- Latest Eq.5 5Y observation in the merged panel: {latest['date'].date().isoformat()}, {latest['x_govt_raw_bps_5y']:.2f} bps.",
        "",
        "## First-Pass Readout",
        "",
        f"- 5Y UST convenience vs 3-6Y marketable supply level correlation: {c_5_supply:.3f}.",
        f"- 10Y UST convenience vs 7-11Y marketable supply level correlation: {c_10_supply:.3f}.",
        f"- 5Y UST convenience vs 3-6Y dealer position level correlation: {c_5_dealer:.3f}.",
        f"- 10Y UST convenience vs 7-11Y dealer position level correlation: {c_10_dealer:.3f}.",
        "",
        "In the supply+dealer+controls level regressions, selected HAC coefficients are:",
        "",
    ]
    for label, row in [
        ("5Y supply bucket", beta_5_supply),
        ("5Y dealer bucket", beta_5_dealer),
        ("10Y supply bucket", beta_10_supply),
        ("10Y dealer bucket", beta_10_dealer),
    ]:
        if row is not None:
            lines.append(f"- {label}: {row['coef_bps_per_1sd']:.2f} bps per 1 SD, t={row['t_stat_hac']:.2f}.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a serious improvement over the FRED-only supply proxy because it uses marketable Treasury supply and maturity buckets.",
            "- It is still not causal. The level relationships are persistent and can be driven by common trends.",
            "- The dealer-capacity variables only start in 2013 in the current API series, so they are better for post-GFC / post-COVID diagnostics than for the full 2000 sample.",
            "- The right next research question is whether same-maturity supply and dealer-position pressure help explain 5Y/10Y UST convenience after controlling for rates, dollar strength, and volatility.",
            "",
            "## Outputs",
            "",
            "- Panel: `output/cleaned/eq5_ust_supply_dealer_official_monthly_panel.csv`",
            "- Treasury supply monthly: `output/cleaned/treasury_supply_official_monthly.csv`",
            "- NY Fed dealer weekly long: `output/cleaned/nyfed_primary_dealer_selected_weekly_long.csv`",
            "- NY Fed dealer monthly: `output/cleaned/nyfed_primary_dealer_selected_monthly.csv`",
            "- Coverage: `output/tables/eq5_supply_dealer_official_variable_coverage.csv`",
            "- Correlations: `output/tables/eq5_supply_dealer_official_correlations.csv`",
            "- Regression models: `output/tables/eq5_supply_dealer_official_regression_models.csv`",
            "- Regression coefficients: `output/tables/eq5_supply_dealer_official_regression_coefficients.csv`",
            f"- Figure: `{figure.relative_to(PROJECT_ROOT)}`",
        ]
    )
    out = REPORT_DIR / "Eq5_Official_Supply_Dealer_Extension_First_Pass.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    coverage.head(30).to_csv(TABLE_DIR / "eq5_supply_dealer_official_top_coverage.csv", index=False)
    return out


def main() -> None:
    for path in [CLEANED_DIR, DATA_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
    coverage = variable_coverage(panel)
    corr = correlation_table(panel)
    coefs, models = regression_tables(panel)
    figure = make_figure(panel)
    memo = write_memo(panel, coverage, corr, coefs, models, figure)

    for path in [
        CLEANED_DIR / "eq5_ust_supply_dealer_official_monthly_panel.csv",
        TABLE_DIR / "eq5_supply_dealer_official_variable_coverage.csv",
        TABLE_DIR / "eq5_supply_dealer_official_correlations.csv",
        TABLE_DIR / "eq5_supply_dealer_official_regression_models.csv",
        TABLE_DIR / "eq5_supply_dealer_official_regression_coefficients.csv",
        figure,
        memo,
    ]:
        print(path)


if __name__ == "__main__":
    main()
