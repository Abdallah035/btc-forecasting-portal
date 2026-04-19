"""
Evaluation metrics — paper's Section 3.4.

Replicates Yenidogan et al. (2023) full metric suite:
MAE, MAPE, MSE, RMSE — plus R² for variance explained.
"""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute the 4 paper metrics + R².

    Parameters
    ----------
    y_true : pd.Series
        Actual prices (test set).
    y_pred : pd.Series
        Model predictions for the same dates.

    Returns
    -------
    dict with keys: MAE, MAPE, MSE, RMSE, R2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse, "R2": r2}
