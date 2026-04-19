"""
Prophet forecasting model — log transform with capped uncertainty.

Inspired by Yenidogan et al. (2023) Section 3.3.2, adapted for our
14-year BTC-USD dataset (vs paper's 2-year BTC-IDR).

KEY ADAPTATIONS FOR OUR DATA:
    1. Log transform: model log(price), exponentiate at the end
       Reason: BTC grew 600x ($200 -> $120k) — linear models can't
       extrapolate that. log(price) only changes by ~6.4x.

    2. Tight changepoint_prior_scale (0.05 default)
       Reason: On log scale, default prior is appropriate. Higher
       values caused exponential blow-up in confidence bands.

    3. Cap weekly_seasonality and yearly_seasonality with low priors
       Reason: BTC has no real weekly cycle; yearly is dominated by
       random news, not pattern. Letting Prophet fit them aggressively
       creates oscillating predictions.

    4. n_changepoints reduced to 25 (Prophet default)
       Reason: With log transform, fewer changepoints prevent overfit.

Prophet model: y(t) = g(t) + s(t) + h(t) + e(t)
"""
import time
from typing import Tuple
import numpy as np
import pandas as pd
from prophet import Prophet


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply natural log to 'y' column."""
    out = df[["ds", "y"]].copy()
    out["y"] = np.log(out["y"])
    return out


def _build_model(
    changepoint_prior_scale: float,
    seasonality_mode: str,
    n_changepoints: int,
    confidence: float,
) -> Prophet:
    """Construct a Prophet model with our standard configuration."""
    return Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        n_changepoints=n_changepoints,
        interval_width=confidence,
        daily_seasonality=False,
        weekly_seasonality=False,   # BTC has no real weekly pattern
        yearly_seasonality=True,
        seasonality_prior_scale=1.0,  # tight — don't over-fit seasons
    )


def train_prophet_auto(
    train_df: pd.DataFrame,
    confidence: float = 0.95,
) -> Tuple[Prophet, float]:
    """
    Train Prophet with sensible defaults for log-scale BTC.

    Auto means: user does not tune, we apply our calibrated defaults.
    """
    start = time.time()
    log_train = _log_transform(train_df)

    model = _build_model(
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        n_changepoints=25,
        confidence=confidence,
    )
    model.fit(log_train)

    elapsed = time.time() - start
    return model, elapsed


def train_prophet_manual(
    train_df: pd.DataFrame,
    changepoint_prior_scale: float = 0.05,
    seasonality_mode: str = "additive",
    n_changepoints: int = 25,
    confidence: float = 0.95,
) -> Tuple[Prophet, float]:
    """
    Train Prophet with user-tuned parameters on log-transformed prices.

    All sliders apply to LOG-SPACE — small values produce stable forecasts.

    Parameters
    ----------
    changepoint_prior_scale : float, default=0.05
        Trend flexibility on log scale. Range: 0.001 - 0.5.
        Stay BELOW 0.1 to avoid exponential blow-up.
    seasonality_mode : {'additive', 'multiplicative'}
        On log-space, 'additive' is the right choice.
    n_changepoints : int, default=25
        Prophet's default. Higher values cause overfitting on log data.
    confidence : float, default=0.95
        Width of uncertainty interval.
    """
    start = time.time()
    log_train = _log_transform(train_df)

    model = _build_model(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        n_changepoints=n_changepoints,
        confidence=confidence,
    )
    model.fit(log_train)

    elapsed = time.time() - start
    return model, elapsed


def make_forecast(
    model: Prophet,
    horizon_days: int,
    freq: str = "D",
    historical_max: float = None,
) -> pd.DataFrame:
    """
    Generate predictions and back-transform from log to real prices.

    Parameters
    ----------
    historical_max : float, optional
        If provided, caps the upper confidence band at 3x this value.
        Prevents exponential blow-up of uncertainty intervals.

    Returns
    -------
    pd.DataFrame
        Columns: ds, yhat, yhat_lower, yhat_upper (real USD prices)
    """
    future = model.make_future_dataframe(periods=horizon_days, freq=freq)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # Back-transform: real_price = exp(log_price)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = np.exp(forecast[col])

    # Cap exploded confidence bands at 3x historical high (sanity check)
    if historical_max is not None:
        cap = historical_max * 3
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(upper=cap)
        forecast["yhat"] = forecast["yhat"].clip(upper=cap)

    return forecast
