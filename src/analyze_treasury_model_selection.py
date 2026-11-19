"""Select low-order ARMA models for weekly changes in 5Y Treasury convenience."""

from itertools import product
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/derived/country_stress_weekly_panel.csv.gz"
OUTPUT = ROOT / "results/tables/treasury_model_selection.csv"
TRAINING_END = pd.Timestamp("2025-01-01")
COHORT = ("AUD", "CAD", "CHF", "DKK", "EUR", "GBP", "NZD", "SEK")
REPORTED_MODELS = ((0, 1, False), (1, 0, False), (0, 1, True), (0, 0, False))


def training_changes() -> pd.Series:
    """Return pre-2025 weekly changes on the existing matched eight-country calendar."""
    source = pd.read_csv(INPUT, parse_dates=["week"])
    country = source.query("tenor == '5y' and week < @TRAINING_END").pivot(
        index="week", columns="currency", values="d_x_govt_raw_bps"
    ).asfreq("W-FRI")
    if set(country.columns) != set(COHORT) or not country.notna().all(axis=1).eq(
        country.notna().any(axis=1)
    ).all():
        raise ValueError("Expected the archived fixed-eight 5Y weekly calendar.")
    return country[list(COHORT)].mean(axis=1, skipna=False)


def estimate_candidates(changes: pd.Series) -> pd.DataFrame:
    """Fit the candidate grid by maximum likelihood and report comparable AIC/AICc."""
    n = changes.count()
    rows = []
    for p, q, drift in product(range(3), range(3), (False, True)):
        if p + q > 3:
            continue
        model = SARIMAX(changes, order=(p, 0, q), trend="c" if drift else "n",
                        enforce_stationarity=True, enforce_invertibility=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", message=r"Non-(stationary|invertible) starting .*",
                                    category=UserWarning)
            fit = model.fit(method="lbfgs", disp=False, maxiter=150)
            if not fit.mle_retvals["converged"] or not np.isfinite(fit.llf):
                fit = model.fit(method="powell", disp=False, maxiter=150)
        if not fit.mle_retvals["converged"] or not np.isfinite(fit.llf):
            raise RuntimeError(f"ARMA({p},{q}) with drift={drift} failed to converge.")
        k = len(fit.params)
        if n <= k + 1:
            raise ValueError("Too few observed weekly changes to calculate AICc.")
        aic = -2 * fit.llf + 2 * k
        rows.append({
            "p": p, "q": q, "allow_drift": drift,
            "start_week": changes.first_valid_index().date().isoformat(),
            "end_week": changes.last_valid_index().date().isoformat(),
            "observed_weeks": n, "parameters": k, "log_likelihood": fit.llf,
            "aic": aic, "aicc": aic + 2 * k * (k + 1) / (n - k - 1),
            "intercept": fit.params.get("intercept"),
            "ar_l1": fit.params.get("ar.L1"), "ar_l2": fit.params.get("ar.L2"),
            "ma_l1": fit.params.get("ma.L1"), "ma_l2": fit.params.get("ma.L2"),
            "innovation_variance_bps2": fit.params["sigma2"],
        })
    return pd.DataFrame(rows).sort_values("aicc", ignore_index=True)


def selection_summary(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.set_index(["p", "q", "allow_drift"]).loc[list(REPORTED_MODELS)]
    return pd.DataFrame({
        "Weekly first-difference model": ["MA(1), no drift", "AR(1), no drift",
                                         "MA(1), with drift", "White noise (level random walk)"],
        "AICc": selected["aicc"].map("{:.2f}".format).to_numpy(),
    })


def main() -> None:
    results = estimate_candidates(training_changes())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(selection_summary(results).to_string(index=False))


if __name__ == "__main__":
    main()
