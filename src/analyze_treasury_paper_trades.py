"""Simulate one-week 5Y bilateral spread exposures using archived forecasts."""

import numpy as np
import pandas as pd

RULES = ("Equal weight", "EWMA inverse volatility", "GARCH inverse volatility",
         "EWMA inverse volatility + filter", "GARCH inverse volatility + filter")


def paper_trade_summary(baseline: pd.DataFrame, risk: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    mean = baseline.query("tenor == '5y' and series != 'Basket'")
    data = mean.merge(risk, on=["origin", "series"], validate="1:1", suffixes=("", "_risk"))
    if not np.allclose(data.actual, data.actual_risk):
        raise ValueError("Mean and volatility forecasts use different realized targets.")
    if not data.groupby("origin").size().eq(8).all():
        raise ValueError("Paper trades require the complete eight-country cohort.")
    future = weekly.query("tenor == '5y'").set_index(["week", "currency"])["d_x_govt_raw_bps"]
    delay_keys = pd.MultiIndex.from_arrays([data.origin + pd.Timedelta(weeks=2), data.series])
    data["delayed"] = future.reindex(delay_keys).to_numpy()
    matched = data.groupby("origin")["delayed"].count()
    if not matched.isin([0, 8]).all():
        raise ValueError("Delay comparison requires common entry weeks across markets.")
    rows = []
    calendar = pd.date_range(data.origin.min(), data.origin.max(), freq="W-FRI")
    rng = np.random.default_rng(18690)
    blocks = rng.integers(0, len(calendar), size=(10000, int(np.ceil(len(calendar) / 4))))
    block_indices = (blocks[:, :, None] + np.arange(4)).reshape(len(blocks), -1)[:, :len(calendar)] % len(calendar)
    for signal, label in (("ma1", "MA(1), no drift"), ("ar1", "AR(1), no drift")):
        for rule, volatility, use_filter in zip(
            RULES, (None, "ewma", "garch", "ewma", "garch"), (False, False, False, True, True), strict=True
        ):
            if volatility is None:
                scale = pd.Series(1 / 8, index=data.index)
            else:
                sigma = np.sqrt(data[volatility])
                inverse = 1 / sigma
                scale = inverse / inverse.groupby(data.origin).transform("sum")
                if use_filter:
                    scale = scale.where(data[signal].abs() / sigma >= 0.25, 0.0)
            weight = np.sign(data[signal]) * scale
            positions = data.assign(gross=weight * data.actual, delay_pnl=weight * data.delayed,
                                    exposure=weight.abs(), trades=weight.ne(0))
            totals = positions.groupby("origin")[["gross", "exposure", "trades"]].sum()
            if (totals.exposure > 1 + 1e-12).any():
                raise ValueError("Position exceeds the DV01 budget.")
            delayed = positions.dropna(subset=["delayed"])
            net25 = totals.gross - .25 * totals.exposure
            equity = np.r_[0.0, net25.cumsum()]
            bootstrap = net25.reindex(calendar, fill_value=0.0).to_numpy()[block_indices].sum(axis=1)
            ci_low, ci_high = np.quantile(bootstrap, [.025, .975])
            rows.append({
                "Signal": label, "Position rule": rule, "Weeks": len(totals),
                "Trades": int(totals.trades.sum()), "Mean DV01 deployed": totals.exposure.mean(),
                **{f"P&L at {c:.2f} bps cost": (totals.gross - c * totals.exposure).sum()
                   for c in (0, .10, .25, .50, 1)},
                "Break-even cost": totals.gross.sum() / totals.exposure.sum(),
                "Max drawdown at 0.25 bps": (np.maximum.accumulate(equity) - equity).max(),
                "Investigative bootstrap lower at 0.25 bps": ci_low,
                "Investigative bootstrap upper at 0.25 bps": ci_high,
                "Delay comparison weeks": delayed.origin.nunique(),
                "Delayed entry P&L at 0.25 bps": (delayed.delay_pnl - .25 * delayed.exposure).sum(),
            })
    return pd.DataFrame(rows)
