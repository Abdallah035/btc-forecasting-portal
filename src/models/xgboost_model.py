"""
XGBoost forecasting model.

Our extension beyond Yenidogan et al. (2023). The paper compared only
ARIMA vs Prophet (statistical models). We add XGBoost (gradient-boosted
trees) to capture non-linear patterns and regime changes that
statistical models miss.

Approach:
    1. Engineer features from price history: lags, rolling stats, date parts
    2. Train XGBoost regressor on log-prices (same reason as ARIMA/Prophet)
    3. Forecast recursively — each prediction feeds the next
"""
import time
from typing import Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# Feature configuration
LAG_DAYS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 30]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from a price series.

    Adds:
        - lag_1, lag_7, lag_14, lag_30      (past prices)
        - rolling_mean_7, rolling_std_7      (week-level patterns)
        - rolling_mean_30, rolling_std_30    (month-level patterns)
        - dow (day of week), month, quarter  (calendar features)

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'ds' (datetime) and 'y' (log-price) columns.

    Returns
    -------
    pd.DataFrame with added feature columns. NaN rows from lag dropped.
    """
    out = df.copy()

    for lag in LAG_DAYS:
        out[f"lag_{lag}"] = out["y"].shift(lag)

    for w in ROLLING_WINDOWS:
        out[f"rolling_mean_{w}"] = out["y"].shift(1).rolling(w).mean()
        out[f"rolling_std_{w}"]  = out["y"].shift(1).rolling(w).std()

    out["dow"] = out["ds"].dt.dayofweek
    out["month"] = out["ds"].dt.month
    out["quarter"] = out["ds"].dt.quarter

    return out.dropna().reset_index(drop=True)


def _feature_columns() -> list:
    """List of column names XGBoost will use as inputs."""
    cols = [f"lag_{l}" for l in LAG_DAYS]
    cols += [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    cols += [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    cols += ["dow", "month", "quarter"]
    return cols


def train_xgboost(
    train_df: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> Tuple[XGBRegressor, float]:
    """
    Train an XGBoost regressor on log-transformed prices with engineered features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must have 'ds' and 'y' columns.
    n_estimators : int, default=300
        Number of boosting rounds (trees).
    max_depth : int, default=5
        Tree depth — deeper = more complex patterns but risks overfit.
    learning_rate : float, default=0.05
        How much each tree contributes. Lower = more stable.

    Returns
    -------
    tuple of (fitted_model, training_time_seconds)
    """
    start = time.time()

    # Log-transform for stability (same reason as ARIMA/Prophet)
    log_df = train_df.copy()
    log_df["y"] = np.log(log_df["y"])

    # Engineer features
    featured = _build_features(log_df)
    feature_cols = _feature_columns()

    X = featured[feature_cols]
    y = featured["y"]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,    # use all CPU cores
        verbosity=0,
    )
    model.fit(X, y)

    elapsed = time.time() - start
    return model, elapsed


def make_xgboost_forecast(
    model: XGBRegressor,
    train_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Recursive multi-step forecast.

    For each future day:
        1. Build features from the history-so-far (real + previously predicted)
        2. Predict the next log-price
        3. Append it to history and repeat

    Returns log-back-transformed predictions in real USD.

    Parameters
    ----------
    model : trained XGBRegressor
    train_df : pd.DataFrame
        Used to seed the recursive loop with real history.
    horizon : int
        Number of future periods to predict.

    Returns
    -------
    pd.DataFrame
        Columns: ds, yhat, yhat_lower, yhat_upper (real USD prices)
        Includes both historical fit AND future predictions.
    """
    feature_cols = _feature_columns()

    # Work in log space throughout
    log_df = train_df.copy()
    log_df["y"] = np.log(log_df["y"])

    # In-sample fit for chart continuity
    featured_train = _build_features(log_df)
    fitted_log = model.predict(featured_train[feature_cols])
    historical_df = pd.DataFrame({
        "ds": featured_train["ds"].values,
        "yhat": np.exp(fitted_log),
    })

    # Compute residual std for confidence bands
    residual_std = (featured_train["y"].values - fitted_log).std()

    # Recursive forecast loop
    history = log_df.copy()
    freq = pd.infer_freq(train_df["ds"]) or "D"
    last_date = train_df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    predictions = []
    for date in future_dates:
        # Append a placeholder row with the date but no y yet
        history = pd.concat([
            history,
            pd.DataFrame({"ds": [date], "y": [np.nan]}),
        ], ignore_index=True)

        # Build features from current history (excluding the placeholder)
        featured = _build_features(history.iloc[:-1])
        if featured.empty:
            predictions.append(np.nan)
            continue

        # Take the most recent feature row (which uses the latest known price as lag-1)
        latest_features = featured.iloc[-1:][feature_cols]

        # Apply same date features for the new date being predicted
        latest_features = latest_features.copy()
        latest_features["dow"] = date.dayofweek
        latest_features["month"] = date.month
        latest_features["quarter"] = date.quarter

        pred_log = model.predict(latest_features)[0]

        # Fill the placeholder so the next iteration uses this prediction as a lag
        history.loc[history.index[-1], "y"] = pred_log
        predictions.append(pred_log)

    # Build forecast DataFrame with confidence bands from residual std
    pred_arr = np.array(predictions)
    z = 1.96  # 95% confidence
    future_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.exp(pred_arr),
        "yhat_lower": np.exp(pred_arr - z * residual_std),
        "yhat_upper": np.exp(pred_arr + z * residual_std),
    })

    # Add confidence bands to historical fit (constant width based on residual)
    historical_df["yhat_lower"] = historical_df["yhat"] * np.exp(-z * residual_std)
    historical_df["yhat_upper"] = historical_df["yhat"] * np.exp(z * residual_std)

    return pd.concat([historical_df, future_df], ignore_index=True)
