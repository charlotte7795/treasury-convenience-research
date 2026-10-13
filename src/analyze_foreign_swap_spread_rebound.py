"""Same-sample accounting for foreign swap spreads in the recent UST rebound."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analyze_treasury_interim import CLEAN, COHORT, FIGURE, REPORT, TABLE, plt

LEGS = ["foreign_swap", "foreign_govt_yield", "usd_swap", "usd_govt_yield"]
METRICS = [f"{leg}_bps" for leg in LEGS] + [
    "foreign_swap_spread_bps", "us_swap_spread_bps", "x_rf_raw_bps", "x_govt_raw_bps"]
WINDOWS = {"2023 average": ("2023-01-31", "2023-12-31"),
           "2023 H2": ("2023-07-31", "2023-12-31"),
           "2024 average": ("2024-01-31", "2024-12-31")}
BIS_URL = "https://www.bis.org/publications/negative-interest-rate-swap-spreads-signal-pressure-government-debt-absorption"
ECB_URL = "https://www.ecb.europa.eu/press/blog/date/2024/html/ecb.blog241114~6a3182c0bd.en.html"


def country_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    assert not raw.duplicated(["date", "currency", "tenor"]).any()
    raw = raw.copy()
    for leg in LEGS:
        raw[f"{leg}_bps"] = 100 * raw[leg]
    assert raw[METRICS].notna().all().all(), "All legs must use the same daily sample"
    raw["month"] = raw.date.dt.to_period("M").dt.to_timestamp("M")
    groups = raw.groupby(["month", "currency", "tenor"])
    monthly = groups[METRICS].mean().join(groups.size().rename("n_days"))
    monthly = monthly.join(groups.regime.nunique().rename("n_regimes")).reset_index()
    monthly.loc[monthly.n_days.lt(15) | monthly.n_regimes.ne(1), METRICS] = np.nan
    return monthly


def contrast(monthly: pd.DataFrame, start: str, stop: str) -> pd.DataFrame:
    baseline = monthly[monthly.month.between(start, stop)]
    latest = monthly[monthly.month.between("2026-01-31", "2026-05-31")]
    keys = ["currency", "tenor"]
    before = baseline.groupby(keys)[METRICS].mean()
    after = latest.groupby(keys)[METRICS].mean()
    delta = after - before
    delta["baseline_months"] = baseline.groupby(keys).x_govt_raw_bps.count()
    delta["latest_months"] = latest.groupby(keys).x_govt_raw_bps.count()
    delta["leg_bridge_error_bps"] = (delta.foreign_swap_spread_bps
                                    - delta.foreign_swap_bps + delta.foreign_govt_yield_bps)
    delta["eq6_bridge_error_bps"] = (delta.x_govt_raw_bps - delta.x_rf_raw_bps
                                    - delta.us_swap_spread_bps + delta.foreign_swap_spread_bps)
    delta["without_foreign_spread_change_bps"] = delta.x_rf_raw_bps + delta.us_swap_spread_bps
    return delta.reset_index()


def instrument_dictionary(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = pd.read_csv(TABLE / "downloaded_bloomberg_ticker_coverage.csv").fillna("")
    # Some blocks lack supplemental metadata. Prefer the fully described instance.
    coverage = coverage.sort_values("field", ascending=False).drop_duplicates("variable")
    lookup = coverage.set_index("variable")
    rows, audit = [], []
    for (ccy, tenor), group in raw.groupby(["currency", "tenor"]):
        regimes = group.regime.unique()
        assert len(regimes) == 1, "Recent comparison must not cross a benchmark splice"
        govt = coverage[coverage.variable.str.startswith(f"{ccy.lower()}_{regimes[0]}_govt_{tenor}_")]
        assert len(govt) == 1
        variables = [group.swap_variable.iloc[0], group.basis_variable.iloc[0], govt.variable.iloc[0]]
        if group.tenor_basis_variable.notna().any():
            variables.append(group.tenor_basis_variable.dropna().iloc[0])
        assert all(group[col].nunique(dropna=True) <= 1 for col in
                   ["swap_variable", "basis_variable", "tenor_basis_variable"])
        audit.append({"currency": ccy, "tenor": tenor, "regime": regimes[0],
                      "n_regimes": len(regimes), "first_date": group.date.min(),
                      "last_date": group.date.max(), "swap_variable": variables[0],
                      "government_variable": variables[2],
                      "floating_benchmark": lookup.loc[variables[0], "floating_leg"]})
        for variable in variables:
            row = lookup.loc[variable]
            rows.append({"currency": ccy, "tenor": tenor, "variable": variable,
                         "ticker": row.ticker, "field": "PX_LAST", "type": row["type"],
                         "unit": row.unit_guess, "floating_leg": row.floating_leg,
                         "name": row["name"], "source": row.source})
    for tenor in ["5y", "10y"]:
        for kind in ["irs", "govt"]:
            candidates = coverage[coverage.variable.str.startswith(f"usd_sofr_{kind}_{tenor}_")]
            assert len(candidates) == 1
            row = candidates.iloc[0]
            rows.append({"currency": "USD", "tenor": tenor, "variable": row.variable,
                         "ticker": row.ticker, "field": "PX_LAST", "type": row["type"],
                         "unit": row.unit_guess, "floating_leg": row.floating_leg,
                         "name": row["name"], "source": row.source})
    return pd.DataFrame(rows).drop_duplicates("ticker"), pd.DataFrame(audit)


def make_figures(monthly: pd.DataFrame, changes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for tenor, ax in zip(["5y", "10y"], axes):
        sub = changes[changes.tenor.eq(tenor)].set_index("currency").reindex(COHORT)
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub.foreign_swap_bps, 0.36, color="#246b91", label="Swap rate change")
        ax.bar(x + 0.18, -sub.foreign_govt_yield_bps, 0.36, color="#ab6c18", label="Minus govt yield change")
        ax.scatter(x, sub.foreign_swap_spread_bps, color="#222222", zorder=3, label="Net spread change")
        ax.axhline(0, color="#444444", lw=0.7)
        ax.set_xticks(x, COHORT)
        ax.set_title(tenor.upper())
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Change / contribution (bps)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Foreign swap spread: 2023 average to 2026 Jan-May average")
    fig.tight_layout()
    fig.savefig(FIGURE / "foreign_swap_spread_country_legs.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)
    for tenor, ax in zip(["5y", "10y"], axes):
        aggregate = monthly[monthly.tenor.eq(tenor)].groupby("month")[METRICS].mean()
        centered = aggregate - aggregate.loc["2023"].mean()
        for metric, label, color, style in [
            ("foreign_swap_bps", "Foreign swap rate", "#246b91", "--"),
            ("foreign_govt_yield_bps", "Foreign govt yield", "#ab6c18", ":"),
            ("foreign_swap_spread_bps", "Foreign swap spread", "#222222", "-")]:
            ax.plot(centered.index, centered[metric], label=label, color=color, ls=style, lw=1.5)
        ax.set_ylabel(f"{tenor.upper()} change (bps)")
        ax.axhline(0, color="#444444", lw=0.7)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_title("Fixed-eight country means, relative to the 2023 average")
    axes[-1].set_xlim(centered.index.min(), centered.index.max())
    axes[-1].set_xticks(pd.to_datetime(["2023-01-31", "2024-01-31", "2025-01-31", "2026-01-31", "2026-05-31"]),
                       ["2023-01", "2024-01", "2025-01", "2026-01", "2026-05"])
    fig.tight_layout()
    fig.savefig(FIGURE / "foreign_swap_spread_monthly_legs.png", dpi=170)
    plt.close(fig)


def main() -> None:
    raw = pd.read_csv(CLEAN / "raw_g10_benchmark_eq6_5y10y_panel.csv", parse_dates=["date"], low_memory=False)
    raw = raw[raw.is_selected_regime_by_public_break & raw.currency.isin(COHORT)
              & raw.date.between("2023-01-01", "2026-05-31")].copy()
    monthly = country_monthly(raw)
    assert monthly[METRICS].notna().all().all(), "All fixed-cohort months must meet coverage rules"
    expected = pd.date_range("2023-01-31", "2026-05-31", freq="ME")
    assert set(zip(monthly.currency, monthly.tenor)) == {(c, t) for c in COHORT for t in ["5y", "10y"]}
    assert monthly.groupby(["currency", "tenor"]).size().eq(len(expected)).all()
    assert set(monthly.month) == set(expected)
    changes = contrast(monthly, *WINDOWS["2023 average"])
    sensitivity = []
    for label, (start, stop) in WINDOWS.items():
        delta = contrast(monthly, start, stop)
        for tenor, group in delta.groupby("tenor"):
            sensitivity.append({"baseline": label, "excluded_currency": "none", "tenor": tenor,
                                "n_currencies": len(group), "negative_foreign_spreads": int(group.foreign_swap_spread_bps.lt(0).sum()),
                                **group[METRICS + ["without_foreign_spread_change_bps"]].mean().to_dict()})
        if label == "2023 average":
            for ccy in COHORT:
                for tenor, group in delta[delta.currency.ne(ccy)].groupby("tenor"):
                    sensitivity.append({"baseline": label, "excluded_currency": ccy, "tenor": tenor,
                                        "n_currencies": len(group), "negative_foreign_spreads": int(group.foreign_swap_spread_bps.lt(0).sum()),
                                        **group[METRICS + ["without_foreign_spread_change_bps"]].mean().to_dict()})
    sensitivity = pd.DataFrame(sensitivity)
    assert sensitivity.foreign_swap_spread_bps.lt(0).all()
    assert sensitivity.x_govt_raw_bps.gt(0).all()
    assert changes[["leg_bridge_error_bps", "eq6_bridge_error_bps"]].abs().max().max() < 1e-8
    prior = pd.read_csv(TABLE / "treasury_interim_country_rebound.csv")
    check = changes.merge(prior, on=["currency", "tenor"], validate="one_to_one")
    assert np.allclose(check.x_govt_raw_bps, check.x_govt_raw_bps_change_bps, atol=1e-8)
    inputs, audit = instrument_dictionary(raw)
    monthly.to_csv(CLEAN / "foreign_swap_spread_country_monthly.csv", index=False)
    for frame, name in [(changes, "country_changes"), (sensitivity, "sensitivity"), (audit, "instrument_audit"), (inputs, "refresh_inputs")]:
        frame.to_csv(TABLE / f"foreign_swap_spread_{name}.csv", index=False)
    (TABLE / "foreign_swap_spread_refresh_inputs.json").write_text(inputs.to_json(orient="records", indent=2) + "\n")
    make_figures(monthly, changes)
    primary = sensitivity[sensitivity.baseline.eq("2023 average") & sensitivity.excluded_currency.eq("none")]
    selected = ["currency", "tenor", "foreign_swap_bps", "foreign_govt_yield_bps", "foreign_swap_spread_bps", "x_govt_raw_bps"]
    memo = ["# Foreign Swap Spreads and the Treasury-Convenience Rebound", "", "Reviewed: 2026-10-03", "",
            "## Result", "",
            "Fixed-eight, same-day raw legs, 2023 annual average versus 2026 January-May average. Monthly means require 15 quotes; all 16 country-tenor pairs have all 41 months. No fill, no recent benchmark splice. Data were not refreshed beyond June 3, 2026.", "",
            "At 5Y the foreign swap rate falls 53.18 bps, while foreign government yields fall only 15.32 bps. Their difference falls 37.86 bps. At 10Y swaps fall 17.07 bps while government yields rise 22.62 bps, lowering the spread 39.69 bps. These are accounting changes, not separately identified policy or supply effects.", "",
            "If the foreign-spread term is held unchanged while the other measured Eq.6 components change, 5Y convenience falls 17.37 bps instead of rebounding 20.49; 10Y falls 28.99 instead of rising 10.70. This is a ceteris-paribus accounting exercise, not an equilibrium counterfactual.", "",
            "Germany/EUR 5Y combines a 42.60-bps swap decline with an 11.16-bps government-yield increase. UK/GBP 5Y combines a 30.90-bps swap decline with a 19.93-bps government-yield increase. Switzerland/CHF has falling yields on both legs, with swaps falling more. There is no single absolute-yield story shared by every currency.", "",
            "## Country Accounting (bps)", "", changes[selected].round(3).to_markdown(index=False), "",
            "## Endpoint and Composition Sensitivity", "",
            sensitivity[["baseline", "excluded_currency", "tenor", "negative_foreign_spreads", "foreign_swap_spread_bps", "x_govt_raw_bps"]].round(3).to_markdown(index=False), "",
            "## Interpretation and Literature", "",
            f"[BIS, December 2024]({BIS_URL}) already documents falling international swap spreads. It discusses cash-bond absorption, repo/balance-sheet costs and demand to receive fixed in swaps as distinct channels. This supports a hypothesis list, not causal attribution in our sample. The possible contribution here is their effect on relative, currency-hedged UST convenience, not discovering negative swap spreads.", "",
            f"[ECB, November 2024]({ECB_URL}) also studies how expected QT affects OIS term premia. Therefore QT cannot be assumed to move only the government-yield leg. Both swap and government rates must remain in a mechanism test.", "",
            "Our foreign benchmarks differ: AUD, DKK, NZD and SEK use local term bank rates; CAD, CHF, EUR and GBP use overnight benchmarks. NZD also needs a tenor-basis adjustment. A SOFR-regime label refers to the USD/basis construction, not to every foreign rate being OIS. The instrument audit retains actual floating benchmarks. Fixed variables rule out a splice explanation but do not establish quote quality or cause.", "",
            "## Data Decision", "",
            f"No new Bloomberg data are required for this accounting result. An optional refresh contains {len(inputs)} previously downloaded, non-empty PX_LAST tickers for the fixed eight currencies and USD, 5Y/10Y, from 2023 to the download date. Refresh extends the endpoint; it does not identify a mechanism.", "",
            "For the next mechanism test focus on German and UK nominal bond outstanding/net issuance and central-bank sovereign holdings. Government stock minus comparable central-bank holdings approximates investor-absorbable stock; adjust units, valuation, eligibility and maturity coverage before subtraction. Fiscal agencies and ECB/BoE publish these series, so a new Bloomberg entitlement is not the first dependency.", "",
            "Incremental Bloomberg checks, if available, are matched 5Y/10Y government curve alternatives, historical IRS PX_BID/PX_ASK, and clearly defined government-collateral repo rates. Confirm benchmark, tenor, quote timestamp, units and historical field availability at DES/FLDS; no unverified tickers or unsupported fields are inserted as guaranteed pulls. A generic benchmark bond is a robustness proxy, not the same estimated constant-maturity curve.", "",
            "Before a causal claim specify whether we are testing bond absorption or swap hedging demand, and obtain a defensible innovation rather than regressing two trending stocks. German/UK issuance and QT are endogenous. The current result is a narrower, well-validated descriptive extension to the replication."]
    (REPORT / "Foreign_Swap_Spread_Rebound_Memo.md").write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(primary.round(3).to_string(index=False))
    print("Refresh tickers:", len(inputs), "monthly country-tenor observations:", len(monthly))


if __name__ == "__main__":
    main()
