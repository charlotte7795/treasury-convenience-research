"""Test stationarity of the archived monthly Treasury convenience series."""

from pathlib import Path
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/derived/treasury_final_country_monthly.csv"
OUTPUT = ROOT / "results/tables/treasury_stationarity_tests.csv"
COHORT = ("AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "NZD", "SEK")
TENORS = ("5y", "10y")
MAX_ADF_LAG = 4


def stationarity_tests() -> pd.DataFrame:
    """Return main tests and fixed-lag trend ADF sensitivity from the archived panel."""
    panel = pd.read_csv(INPUT, parse_dates=["month"])
    months = pd.period_range("2023-01", "2026-09", freq="M")
    panel["month"] = panel["month"].dt.to_period("M")
    rows = []
    for tenor in TENORS:
        levels = panel.loc[panel["tenor"].eq(tenor)].pivot(
            index="month", columns="currency", values="x_govt_raw_bps"
        )
        if set(levels.columns) != set(COHORT) or not levels.index.equals(months) or levels.isna().any().any():
            raise ValueError(f"{tenor} monthly panel must contain the complete fixed-eight cohort.")
        levels["basket"] = levels[list(COHORT)].mean(axis=1)
        for market in ("basket", *COHORT):
            level = levels[market]
            specs = [("level", "c", level), ("level_with_trend", "ct", level),
                     ("first_difference", "c", level.diff().dropna())]
            for transform, deterministic, values in specs:
                base = {
                    "tenor": tenor, "market": market, "transformation": transform,
                    "deterministic": deterministic, "start_month": str(values.index[0]),
                    "end_month": str(values.index[-1]), "n_months": len(values),
                }
                for fixed_lag in ([None, *range(MAX_ADF_LAG + 1)]
                                  if market == "basket" and deterministic == "ct" else [None]):
                    statistic, p, lag, n_effective, *_ = adfuller(
                        values,
                        maxlag=MAX_ADF_LAG if fixed_lag is None else fixed_lag,
                        regression=deterministic,
                        autolag="AIC" if fixed_lag is None else None,
                    )
                    rows.append({**base, "test": "ADF", "statistic": statistic,
                                 "p_value": p, "p_value_bound": "=", "lag": lag,
                                 "lag_selection": "AIC" if fixed_lag is None else "fixed",
                                 "effective_n": n_effective})
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", InterpolationWarning)
                    statistic, p, lag, _ = kpss(values, regression=deterministic, nlags="auto")
                boundary = any(issubclass(w.category, InterpolationWarning) for w in caught)
                if boundary and p not in (0.01, 0.10):
                    raise ValueError(f"Unexpected KPSS p-value bound: {p}")
                bound = ("<=" if p == 0.01 else ">=") if boundary else "="
                rows.append({**base, "test": "KPSS", "statistic": statistic, "p_value": p,
                             "p_value_bound": bound, "lag": lag,
                             "lag_selection": "auto", "effective_n": len(values)})
    return pd.DataFrame(rows)


def basket_summary(tests: pd.DataFrame) -> pd.DataFrame:
    """Build the presentation table from the main basket test results."""
    main = tests.loc[tests["market"].eq("basket") & tests["lag_selection"].ne("fixed")]

    def p_text(row: pd.Series) -> str:
        if row.p_value_bound == "<=":
            return f"≤{row.p_value:.2f}"
        if row.p_value_bound == ">=":
            return f"≥{row.p_value:.2f}"
        return "<0.001" if row.p_value < 0.001 else f"{row.p_value:.3f}"

    summary = []
    for tenor in TENORS:
        rows = main.loc[main["tenor"].eq(tenor)].set_index(["transformation", "test"])
        adf_diff = p_text(rows.loc[("first_difference", "ADF")])
        kpss_diff = p_text(rows.loc[("first_difference", "KPSS")])
        trend_adf = rows.loc[("level_with_trend", "ADF")]
        summary.append({
            "Equal-weight eight-country basket": tenor.upper(),
            "Level ADF p": p_text(rows.loc[("level", "ADF")]),
            "Level KPSS p": p_text(rows.loc[("level", "KPSS")]),
            "Trend ADF p": p_text(trend_adf) if tenor == "5y" else f"{trend_adf.p_value:.4f}",
            "First difference": f"ADF p{adf_diff}, KPSS p{kpss_diff}",
        })
    return pd.DataFrame(summary)


def main() -> None:
    tests = stationarity_tests()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tests.to_csv(OUTPUT, index=False)
    print(basket_summary(tests).to_string(index=False))


if __name__ == "__main__":
    main()
