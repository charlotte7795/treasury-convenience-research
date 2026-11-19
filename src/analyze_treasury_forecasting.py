"""One-week 5Y/10Y Treasury convenience forecasts and 5Y volatility diagnostics."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.analyze_treasury_model_selection import COHORT, INPUT, TRAINING_END

TABLES = Path(__file__).resolve().parents[1] / "results/tables"
TARGET = "d_x_govt_raw_bps"
BASE = {"rw": "Random walk", "ma1": "MA(1), no drift", "ar1": "AR(1), no drift"}
EXTRA = {
    "foreign_spread": ["d_foreign_swap_spread_bps"],
    "foreign_legs": ["d_foreign_govt_yield_bps", "d_foreign_swap_bps"],
    "vix": ["stress_dlog_vix"],
}
EXTRA_LABELS = {
    "foreign_spread": "Regression + foreign swap spread",
    "foreign_legs": "Regression + foreign yield and swap rate",
    "vix": "Regression + Δlog(VIX)",
}
WEEK = pd.Timedelta(weeks=1)


def series_panels(weekly: pd.DataFrame, tenor: str, columns: list[str]) -> dict:
    data = weekly.loc[weekly["tenor"].eq(tenor)]
    countries = {
        key: d.set_index("week")[columns].sort_index().asfreq("W-FRI")
        for key, d in data.groupby("currency")
    }
    if set(countries) != set(COHORT):
        raise ValueError("Unexpected weekly currency cohort.")
    calendar = countries[COHORT[0]].index
    if not all(panel.index.equals(calendar) for panel in countries.values()):
        raise ValueError("Weekly currency calendars do not match.")
    valid = pd.DataFrame({k: v[TARGET].notna() for k, v in countries.items()})
    if not valid.all(axis=1).eq(valid.any(axis=1)).all():
        raise ValueError("The fixed cohort must share the same eligible weeks.")
    return {**countries, "Basket": sum(countries.values()) / len(COHORT)}


def fit_parameter(changes: pd.Series, order: tuple, start=None):
    model = SARIMAX(changes, order=order, trend="n", enforce_stationarity=True,
                    enforce_invertibility=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = model.fit(start_params=start, method="lbfgs", disp=False, maxiter=150)
        if not fit.mle_retvals["converged"] or not np.isfinite(fit.llf):
            fit = model.fit(method="powell", disp=False, maxiter=150)
    if not fit.mle_retvals["converged"] or not np.isfinite(fit.llf):
        raise RuntimeError(f"Weekly ARMA{order} did not converge.")
    return fit


def ma1_projection(changes: pd.Series, theta: float) -> float:
    """Solve the classroom covariance projection on the observed calendar positions."""
    values = changes.to_numpy()
    observed = np.flatnonzero(np.isfinite(values))
    lag = np.abs(observed[:, None] - observed)
    covariance = np.where(lag == 0, 1 + theta ** 2, np.where(lag == 1, theta, 0.0))
    cross_covariance = np.where(len(values) - observed == 1, theta, 0.0)
    return float(solve(covariance, cross_covariance, assume_a="pos") @ values[observed])


def blp_forecasts(changes: pd.Series) -> pd.DataFrame:
    """Refit no-drift ARMA parameters through each origin and project one week ahead."""
    observed = pd.DataFrame({"current": changes, "actual": changes.shift(-1)}).dropna()
    rows = []
    models = {"ma1": ((0, 0, 1), "ma.L1"), "ar1": ((1, 0, 0), "ar.L1")}
    params = {key: fit_parameter(changes.loc[changes.index < TRAINING_END], order).params
              for key, (order, _) in models.items()}
    for origin, point in observed.loc[TRAINING_END:].iterrows():
        history = changes.loc[:origin]
        row = {"origin": origin, "actual": point.actual, "rw": 0.0}
        for key, (order, param_name) in models.items():
            fit = fit_parameter(history, order, params[key])
            params[key] = fit.params
            coefficient = fit.params[param_name]
            row[key] = ma1_projection(history, coefficient) if key == "ma1" else coefficient * point.current
            if not np.isclose(row[key], fit.forecast(1).iloc[0], atol=1e-8, rtol=1e-7):
                raise RuntimeError("Covariance projection and state-space prediction disagree.")
        rows.append(row)
    return pd.DataFrame(rows)


def augmented_forecasts(panel: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Attach three separate direct one-step regressions to a common forecast sample."""
    data = panel.assign(actual=panel[TARGET].shift(-1)).dropna()
    rows = []
    for origin, observed in data.loc[TRAINING_END:].iterrows():
        training = data.loc[:origin - WEEK]
        row = {"origin": origin}
        for name, extra_columns in EXTRA.items():
            columns = [TARGET, *extra_columns]
            x = np.column_stack([np.ones(len(training)), training[columns]])
            beta, _, rank, _ = np.linalg.lstsq(x, training["actual"], rcond=None)
            if rank != x.shape[1]:
                raise ValueError(f"Rank-deficient {name} forecast at {origin}.")
            row[name] = beta @ np.r_[1.0, observed[columns].astype(float)]
        rows.append(row)
    return baseline.merge(pd.DataFrame(rows), on="origin", validate="1:1")


def garch_filter(squared, omega, alpha, beta, initial):
    predicted = np.empty(len(squared))
    variance = initial
    for t, value in enumerate(squared):
        predicted[t] = variance
        innovation = value if np.isfinite(value) else variance
        variance = omega + alpha * innovation + beta * variance
    return predicted


def estimate_garch(series):
    squared = series.to_numpy() ** 2
    scale = np.nanmean(squared)
    squared = squared / scale
    observed = np.isfinite(squared)

    def nll(params):
        p = garch_filter(squared, *params, initial=1.0)[observed]
        return 0.5 * np.sum(np.log(p) + squared[observed] / p)

    options = {"maxiter": 150, "ftol": 1e-10}
    fits = [
        minimize(nll, start, method="SLSQP",
                 bounds=[(1e-8, 10.0), (0, .999), (0, .999)],
                 constraints={"type": "ineq", "fun": lambda p: .999 - p[1:].sum()},
                 options=options)
        for start in ((.05, .05, .90), (.5, .2, .3), (1., 0., 0.))
    ]
    best = min((fit for fit in fits if fit.success), key=lambda fit: fit.fun)
    omega, alpha, beta = best.x
    assert alpha + beta <= .999001
    assert best.fun <= nll(np.array([1., 0., 0.])) + 1e-7
    return np.array([omega * scale, alpha, beta]), scale


def variance_forecasts(changes, centers):
    # Random walk forecasts the change at zero, so archived changes are its errors.
    initial = changes.loc[changes.index < TRAINING_END]
    params, scale = estimate_garch(initial)
    squared = changes.pow(2)
    garch = garch_filter(squared.to_numpy(), *params, initial=scale)
    ewma = garch_filter(squared.to_numpy(), 0., .06, .94, initial=scale)
    bank = pd.DataFrame({"constant": squared.expanding(min_periods=20).mean(),
                         "ewma": pd.Series(ewma, index=changes.index).shift(-1),
                         "garch": pd.Series(garch, index=changes.index).shift(-1)})
    prediction = centers[["origin", "actual"]].merge(bank, left_on="origin", right_index=True,
                                                     validate="1:1")
    assert prediction[["constant", "ewma", "garch"]].gt(0).all().all()
    return prediction, params


def location_scores(rows, models):
    error = rows[models].sub(rows.actual, axis=0)
    signs = np.sign(rows[models]).eq(np.sign(rows.actual), axis=0)
    valid = rows[models].ne(0).mul(rows.actual.ne(0), axis=0)
    return pd.DataFrame({"rmse": error.pow(2).mean().pow(.5), "mae": error.abs().mean(),
                         "hit": signs.where(valid).mean(), "hit_n": valid.sum()})


def uncertainty_scores(rows):
    models = ["constant", "ewma", "garch"]
    var = rows[models]
    abs_e = rows.actual.abs().to_numpy()[:, None]
    return pd.DataFrame({"qlike": (np.log(var) + var.rdiv(rows.actual.pow(2), axis=0)).mean(),
                         "mean_sigma": np.sqrt(var).mean(),
                         "coverage80": (abs_e <= norm.ppf(.90) * np.sqrt(var)).mean(),
                         "coverage95": (abs_e <= norm.ppf(.975) * np.sqrt(var)).mean()})



def forecast_summary(points: pd.DataFrame, models: dict, by_year: bool) -> pd.DataFrame:
    rows = []
    for tenor, subset in points.groupby("tenor", sort=False):
        for sample, is_basket in (("Basket", True), ("Bilateral spreads", False)):
            frame = subset.loc[subset["series"].eq("Basket").eq(is_basket)]
            periods = [(str(y), g) for y, g in frame.groupby(frame["origin"].dt.year)] if by_year else []
            for period, sample_rows in [*periods, ("All", frame)]:
                scores = location_scores(sample_rows, list(models))
                rows.append({"Tenor": tenor.upper(), "Sample": sample, "Period": period,
                             "Forecasts": len(sample_rows),
                             **{label: scores.loc[key, "rmse"] for key, label in models.items()}})
    return pd.DataFrame(rows)


def volatility_summary(risk: pd.DataFrame) -> pd.DataFrame:
    basket = uncertainty_scores(risk.loc[risk["series"].eq("Basket")])
    pairs = uncertainty_scores(risk.loc[risk["series"].ne("Basket")])
    return pd.DataFrame({
        "Volatility model": ["Expanding historical variance", "EWMA", "GARCH(1,1)"],
        "Basket QLIKE": basket.qlike.to_numpy(), "Bilateral QLIKE": pairs.qlike.to_numpy(),
        "Basket 80% coverage": basket.coverage80.to_numpy(),
        "Basket 95% coverage": basket.coverage95.to_numpy(),
        "Bilateral 80% coverage": pairs.coverage80.to_numpy(),
        "Bilateral 95% coverage": pairs.coverage95.to_numpy(),
    })


def run_forecasts(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [TARGET, *(c for features in EXTRA.values() for c in features)]
    baseline, augmented, risk = [], [], []
    for tenor in ("5y", "10y"):
        panels = series_panels(weekly, tenor, cols if tenor == "5y" else [TARGET])
        for series, panel in panels.items():
            forecasts = blp_forecasts(panel[TARGET])
            baseline.append(forecasts.assign(tenor=tenor, series=series))
            if tenor == "5y":
                augmented.append(augmented_forecasts(panel, forecasts).assign(tenor=tenor, series=series))
                variance, _ = variance_forecasts(panel[TARGET], forecasts)
                risk.append(variance.assign(series=series))
    return tuple(pd.concat(frames, ignore_index=True) for frames in (baseline, augmented, risk))


def main() -> None:
    from src.analyze_treasury_paper_trades import paper_trade_summary

    weekly = pd.read_csv(INPUT, parse_dates=["week"])
    baseline, augmented, risk = run_forecasts(weekly)
    tables = {
        "treasury_one_week_blp_rmse.csv": forecast_summary(baseline, BASE, by_year=True),
        "treasury_one_week_augmented_rmse.csv": forecast_summary(augmented, {**BASE, **EXTRA_LABELS}, by_year=False),
        "treasury_one_week_volatility.csv": volatility_summary(risk),
        "treasury_one_week_paper_trades.csv": paper_trade_summary(baseline, risk, weekly),
    }
    for filename, table in tables.items():
        table.to_csv(TABLES / filename, index=False)
        print(f"{filename}\n{table.round(4).to_string(index=False)}\n")


if __name__ == "__main__":
    main()
