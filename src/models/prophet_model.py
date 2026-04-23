"""
Prophet forecasting model — optimized for BTC based on research.

Optimizations over Yenidogan et al. (2023) defaults:
    1. Log transformation for exponential growth stability
    2. Logistic growth with cap/floor (not linear — BTC has saturation)
    3. changepoint_prior_scale = 0.001 (research-backed for crypto)
    4. Auto-disable yearly seasonality when data is too short
    5. seasonality_prior_scale = 0.1 (tight — prevents overfit to noise)

Research sources:
    - https://medium.com/@alexzap922/btc-price-prediction-using-fb-prophet
    - Facebook Prophet GitHub Issue #797 (high-variance data handling)
    - Facebook Prophet GitHub Issue #859 (logistic vs linear growth)
"""
import time
from typing import Tuple
import numpy as np
import pandas as pd
from prophet import Prophet


def _prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transform and add cap/floor for logistic growth.

    Logistic growth needs:
        - 'cap':   upper saturation (we use 2x historical max)
        - 'floor': lower saturation (0 — BTC cannot go negative)
    """
    out = df[["ds", "y"]].copy()
    out["y"] = np.log(out["y"])
    # In log space, cap is log(2 * real_max), floor is log(0.01) ≈ minimum plausible price
    real_max = np.exp(out["y"].max())
    out["cap"] = np.log(real_max * 2)
    out["floor"] = np.log(0.01)  # $0.01 minimum (practically zero)
    return out


def _has_enough_yearly_data(train_df: pd.DataFrame) -> bool:
    """Check if we have at least 2 full years of data for yearly seasonality."""
    span_days = (train_df["ds"].max() - train_df["ds"].min()).days
    return span_days >= 730  # 2 years


def train_prophet_auto(
    train_df: pd.DataFrame,
    confidence: float = 0.95,
) -> Tuple[Prophet, float]:
    """
    Train Prophet with research-optimized defaults.

    Uses logistic growth on log-transformed prices — the combination that
    research shows works best for exponentially growing volatile assets.
    """
    start = time.time()
    log_train = _prepare_training_data(train_df)

    model = Prophet(
        growth="logistic",                      # saturation-aware growth
        changepoint_prior_scale=0.001,          # research-backed value
        seasonality_prior_scale=0.1,            # tight seasonality
        seasonality_mode="additive",            # on log scale
        interval_width=confidence,
        daily_seasonality=False,
        weekly_seasonality=False,               # BTC has no real weekly cycle
        yearly_seasonality=_has_enough_yearly_data(log_train),
    )
    model.fit(log_train)

    elapsed = time.time() - start
    return model, elapsed


def train_prophet_manual(
    train_df: pd.DataFrame,
    changepoint_prior_scale: float = 0.001,
    seasonality_mode: str = "additive",
    n_changepoints: int = 25,
    confidence: float = 0.95,
) -> Tuple[Prophet, float]:
    """
    Train Prophet with user-tuned parameters.

    Defaults research-backed for BTC. Users can override via sidebar.
    """
    start = time.time()
    log_train = _prepare_training_data(train_df)

    model = Prophet(
        growth="logistic",
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=0.1,
        seasonality_mode=seasonality_mode,
        n_changepoints=n_changepoints,
        interval_width=confidence,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=_has_enough_yearly_data(log_train),
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

    For logistic growth, future dates need cap and floor columns too.
    """
    future = model.make_future_dataframe(periods=horizon_days, freq=freq)

    # Logistic growth requires cap/floor on the future DataFrame
    # We derive them from the model's training state
    if historical_max is not None:
        future["cap"] = np.log(historical_max * 2)
        future["floor"] = np.log(0.01)
    else:
        # Fallback — use the model's own history_ if available
        future["cap"] = model.history["cap"].iloc[0] if "cap" in model.history else np.log(1e7)
        future["floor"] = model.history["floor"].iloc[0] if "floor" in model.history else np.log(0.01)

    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # Back-transform: real_price = exp(log_price)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = np.exp(forecast[col])

    # Cap upper band at 5x historical max to prevent visual explosion
    if historical_max is not None:
        cap = historical_max * 5
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(upper=cap)
        forecast["yhat"] = forecast["yhat"].clip(upper=cap)

    return forecast
