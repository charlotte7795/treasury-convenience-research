"""Normalize the Nov. 3 public-source sovereign-supply collection."""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/official/stress_supply_2026-11-03"
OUT = ROOT / "outputs/samim-stress-supply"
DATES = pd.date_range("2021-12-31", "2026-06-30", freq=pd.offsets.QuarterEnd(startingMonth=12))
COUNTRIES = dict(zip("AUS CAN CHE DNK DEU GBR NZL SWE".split(), "AUD CAD CHF DKK EUR GBP NZD SEK".split()))


def gdp():
    data = pd.read_csv(DATA / "oecd_nominal_gdp.csv")
    data = data.loc[data["ADJUSTMENT"].eq("N")].copy()
    assert len(data) == 176 and data["UNIT_MULT"].eq(6).all()
    assert data["CURRENCY"].eq(data["REF_AREA"].map(COUNTRIES)).all()
    for field, value in {"TRANSACTION": "B1GQ", "PRICE_BASE": "V", "UNIT_MEASURE": "XDC"}.items():
        assert data[field].eq(value).all()
    data["quarter_end"] = pd.PeriodIndex(data["TIME_PERIOD"], freq="Q").end_time.normalize()
    data = data.sort_values(["CURRENCY", "quarter_end"])
    data["gdp_lcu_bn"] = data.groupby("CURRENCY")["OBS_VALUE"].transform(lambda s: s.rolling(4).sum()) / 1000
    return data.pivot(index="quarter_end", columns="CURRENCY", values="gdp_lcu_bn").loc[DATES]


def canada():
    records = json.loads((DATA / "boc_g4.json").read_text())["observations"]
    data = pd.DataFrame(
        (pd.Timestamp(r["d"]) + pd.offsets.MonthEnd(), float(r["V37371"]["v"]), float(r["V37378"]["v"]))
        for r in records
    ).set_index(0).loc[DATES] / 1000
    return data[1] + data[2], data[1]


def new_zealand():
    data = pd.read_excel(DATA / "nzdm_bonds_history.xlsx", sheet_name="Month_end", header=5)
    data = data.loc[data["Type"].eq("Nominal") & data["Month end"].isin(DATES)]
    amounts = data.iloc[:, 6:].apply(pd.to_numeric)
    assert amounts.notna().all().all()
    total, cb, eqc, sresl, market = amounts.columns
    assert (amounts[total] - amounts[eqc] - amounts[sresl] - amounts[cb] - amounts[market]).abs().max() < 1.5
    group = amounts.groupby(data["Month end"]).sum().loc[DATES] / 1000
    return group[market] + group[cb], group[cb]


def sweden():
    issued = pd.read_csv(DATA / "sweden_conventional_stock.csv", parse_dates=["quarter_end"]).set_index("quarter_end").loc[DATES]
    holds = pd.read_csv(DATA / "riksbank_bond_holdings.csv", sep=";")
    conventional = holds["Security ID / Värdepappers ID"].str.fullmatch(r"SO1\d{3}", na=False)
    subset = holds.loc[conventional, DATES.strftime("%Y-%m-%d")]
    assert subset.notna().all().all()
    return issued, pd.Series(subset.sum().to_numpy() / 1e9, index=DATES)


def united_kingdom():
    issued = pd.read_excel(DATA / "uk_dmo_history.xls", sheet_name="Table data (gross)", header=None)
    assert "Conventional" in issued.iloc[4, 0]
    report_dates = pd.to_datetime(issued.iloc[0, 1:], errors="coerce").dt.to_period("Q").dt.end_time.dt.normalize()
    stock = pd.Series(pd.to_numeric(issued.iloc[4, 1:], errors="coerce").to_numpy(), index=report_dates).dropna().loc[DATES]
    nominal = "Total allocation (nominal £mn)"
    ops = []
    for name, sign in (("purchases", 1), ("sales", -1)):
        part = pd.read_excel(DATA / f"boe_apf_{name}.xlsx", header=1)
        part = part.loc[part["ISIN"].notna()].copy()
        part["amount"] = sign * pd.to_numeric(part[nominal])
        ops.append(part)
    trades = pd.concat(ops)
    suffix = trades["Bond\n"].str.extract(r"_(\d{2})(\d{2})(\d{2})$", expand=True)
    assert suffix.notna().all().all()
    trades["maturity"] = pd.to_datetime("20" + suffix[2] + suffix[1] + suffix[0], format="%Y%m%d").to_numpy()
    assert trades.groupby("ISIN")["maturity"].nunique().eq(1).all()
    assert (trades["maturity"] > trades["Settlement date"]).all()

    def remaining(date):
        alive = trades.loc[trades["Settlement date"].le(date) & trades["maturity"].gt(date)]
        by_isin = alive.groupby("ISIN")["amount"].sum()
        assert by_isin.min() > -0.01
        return by_isin.sum() / 1000

    snapshot = pd.read_excel(DATA / "boe_apf.xlsx", header=1)
    official = snapshot.loc[pd.to_datetime(snapshot["Maturity Date"], errors="coerce").notna(), "Total Nominal (£bn)"].sum()
    assert abs(remaining(pd.Timestamp("2026-10-28")) - official) < 0.15
    return stock, pd.Series({date: remaining(date) for date in DATES})


def australia():
    data = pd.read_csv(DATA / "rba_ags_summary.csv", skiprows=10)
    data.index = pd.to_datetime(data["Series ID"], format="%d-%b-%Y") + pd.offsets.MonthEnd(0)
    assert data.index.is_unique
    return data.loc[DATES, "AGSHSTOT"] / 1000


def bis_candidates(archive):
    countries = {"CH": "CHF", "DK": "DKK", "DE": "EUR"}
    series = {f"Q:N:{area}:XW:S1311:S1:N:L:LE:F3:T:_Z:{ccy}:_T:N:V:N:_T": ccy for area, ccy in countries.items()}
    quarter_cols = [str(p).replace("Q", "-Q") for p in DATES.to_period("Q")]
    meta = ["Series", "UNIT_MULT", "Unit", "VALUATION", "MATURITY", "CURRENCY_DENOM"]
    data = pd.read_csv(archive, usecols=lambda c: c in meta or c in quarter_cols)
    data = data.loc[data["Series"].isin(series)].copy()
    assert len(data) == 3 and data["UNIT_MULT"].eq(9).all()
    for key, value in {"VALUATION": "N", "MATURITY": "T", "CURRENCY_DENOM": "_T"}.items():
        assert data[key].eq(value).all(), data[meta]
    data["currency"] = data["Series"].map(series)
    long = data.melt(id_vars=meta + ["currency"], var_name="period", value_name="bis_all_scope_gross_lcu_bn").dropna()
    long["quarter_end"] = pd.PeriodIndex(long["period"], freq="Q").end_time.normalize()
    return long.sort_values(["currency", "quarter_end"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bis-archive", type=Path, help="Official DSS bulk file used to regenerate the candidate extract")
    parser.add_argument("--workbook-input", type=Path, help="Optional JSON handoff to the Excel builder")
    args = parser.parse_args()
    bis_file = DATA / "bis_broad_nominal_candidates.csv"
    if args.bis_archive:
        bis_candidates(args.bis_archive).to_csv(bis_file, index=False)
    bis = pd.read_csv(bis_file, parse_dates=["quarter_end"])
    gdp_values = gdp()
    se_stock, se_holds = sweden()
    pairs = {
        "CAD": canada(),
        "GBP": united_kingdom(),
        "NZD": new_zealand(),
        "SEK": (se_stock["outstanding_lcu_bn"], se_holds),
    }
    au_holds = australia()
    frames = []
    for ccy in sorted(COUNTRIES.values()):
        frame = pd.DataFrame({"currency": ccy, "nominal_gdp_4q_lcu_bn": gdp_values[ccy]}, index=DATES)
        frame[["outstanding_lcu_bn", "cb_held_lcu_bn"]] = float("nan")
        if ccy in pairs:
            frame["outstanding_lcu_bn"], frame["cb_held_lcu_bn"] = pairs[ccy]
        if ccy == "AUD":
            frame["cb_held_lcu_bn"] = au_holds
        frame["outstanding_public_date"] = se_stock["public_date"] if ccy == "SEK" else ""
        frame["outstanding_release_source"] = se_stock["source"] if ccy == "SEK" else ""
        frame["holdings_public_date"] = ""
        frame["gdp_public_date"] = ""
        frame["historical_eligibility"] = "Not verified"
        frame["coverage"] = "Paired" if ccy in pairs else "Holdings only" if ccy == "AUD" else "BIS candidate only"
        if ccy == "GBP":
            frame["coverage"] = "Paired: APF only"
        frames.append(frame.rename_axis("quarter_end").reset_index())
    panel = pd.concat(frames, ignore_index=True).merge(
        bis[["currency", "quarter_end", "bis_all_scope_gross_lcu_bn"]], how="left", validate="1:1",
    )
    panel["net_supply_pct_gdp"] = (panel["outstanding_lcu_bn"] - panel["cb_held_lcu_bn"]) / panel["nominal_gdp_4q_lcu_bn"] * 100
    complete = panel["net_supply_pct_gdp"].notna()
    assert panel.shape[0] == 152 and not panel.duplicated(["currency", "quarter_end"]).any()
    assert panel["nominal_gdp_4q_lcu_bn"].gt(0).all()
    assert complete.sum() == 76 and panel["cb_held_lcu_bn"].count() == 95
    assert panel.loc[complete, "net_supply_pct_gdp"].gt(0).all()
    assert panel["bis_all_scope_gross_lcu_bn"].count() == 54
    panel.to_csv(OUT / "sovereign_stress_supply_public_panel.csv", index=False)
    if args.workbook_input:
        args.workbook_input.write_text(panel.to_json(orient="records", date_format="iso"))
    print(panel.loc[panel["quarter_end"].eq(DATES[-1]), [
        "currency", "outstanding_lcu_bn", "cb_held_lcu_bn", "nominal_gdp_4q_lcu_bn", "net_supply_pct_gdp",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
