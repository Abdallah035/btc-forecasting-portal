"""
ARIMA forecasting model.

Replicates Yenidogan et al. (2023) Section 3.3.1:
- Augmented Dickey-Fuller (ADF) stationarity test
- Auto-selection of (p, d, q) via AIC criterion
- Log-transform for exponential price growth (our adaptation)
"""

import time 
from typing import Tuple , Dict
import numpy as np
import pandas as pd
from  pmdarima import auto_arima 
from statsmodels.tsa.stattools import adfuller

def adf_test(series: pd.Series) -> Dict[str, float]:
    """
    Augmented Dickey-Fuller test for stationarity (paper Section 3.3.1).

    Returns
    -------
    dict with keys:
        - statistic : test statistic
        - p_value : significance (< 0.05 means stationary)
        - is_stationary : bool
        - lags_used : how many lags the test used
    """
    result = adfuller(series.dropna(),autolag = "AIC")
    return {
        "statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
        "lags_used": result[2],
    }


def train_arima(
    train_df: pd.DataFrame,
    seasonal: bool = False,
) -> Tuple[object, float, tuple]:
    """
    Train an ARIMA model with auto-selected (p, d, q) on log-prices.

    Uses pmdarima.auto_arima which automates the paper's manual process:
    - Tests multiple (p, d, q) combinations
    - Selects the one minimizing AIC
    - Internally runs ADF test to determine d

    Parameters
    ----------
    train_df : pd.DataFrame
        Must have 'ds' (datetime) and 'y' (price) columns.
    seasonal : bool, default=False
        Set True for SARIMA. Off by default — slower and rarely helps BTC.

    Returns
    -------
    tuple of (fitted_model, training_time_seconds, order)
        order is the chosen (p, d, q) tuple.
    """
    start = time.time()

    log_y = np.log(train_df["y"].values)

    model = auto_arima(
        log_y,
        seasonal=seasonal,
        stepwise=True,        # smarter search than full grid
        suppress_warnings=True,
        error_action="ignore",
        max_p=5, max_q=5,     # limit search space for speed
        max_d=2,
        information_criterion="aic",  # paper uses AIC
    )

    elapsed = time.time() - start
    return model, elapsed, model.order


def make_arima_forecast(
    model,
    train_df: pd.DataFrame,
    horizon: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Generate ARIMA predictions and back-transform from log to USD.

    Parameters
    ----------
    model : trained pmdarima model
    train_df : pd.DataFrame
        The training data (used to get the last date and historical fit).
    horizon : int
        Number of future periods to forecast.
    confidence : float, default=0.95
        Width of confidence interval.

    Returns
    -------
    pd.DataFrame
        Columns: ds, yhat, yhat_lower, yhat_upper (real USD prices)
        Includes both historical fit AND future predictions.
    """
    # Future predictions with confidence intervals
    pred_log, conf_int_log = model.predict(
        n_periods=horizon,
        return_conf_int=True,
        alpha=1 - confidence,
    )

    # Generate future dates
    last_date = train_df["ds"].iloc[-1]
    freq = pd.infer_freq(train_df["ds"]) or "D"
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    # Build forecast DataFrame in log-space, then exponentiate
    future_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.exp(pred_log),
        "yhat_lower": np.exp(conf_int_log[:, 0]),
        "yhat_upper": np.exp(conf_int_log[:, 1]),
    })

    # Historical fit (in-sample predictions for the chart)
    fitted_log = model.predict_in_sample()
    historical_df = pd.DataFrame({
        "ds": train_df["ds"].values,
        "yhat": np.exp(fitted_log),
        "yhat_lower": np.exp(fitted_log) * 0.95,  # rough band for visual continuity
        "yhat_upper": np.exp(fitted_log) * 1.05,
    })

    return pd.concat([historical_df, future_df], ignore_index=True)