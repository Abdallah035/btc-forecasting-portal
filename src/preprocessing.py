"""
Preprocessing utilities — replicates Yenidogan et al. (2023) Section 3.3.

Only includes paper features applicable to our BTC-USD dataset:
- 80/20 chronological train/test split
- Multi-granularity resampling (Daily / Weekly / Monthly) — paper Section 3.1
- Moving averages for technical indicator overlays (rubric requirement)
"""

from typing import Tuple
import pandas as pd

def train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame chronologically into 80% train / 20% test.

    Replicates Yenidogan et al. (2023) Section 3.3(paper i read):
        "the dataset is split into combination of 80% training dataset
         and 20% testing dataset."

    The split is purely chronological (no shuffling) — required for
    time series to prevent leaking future information into training.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered DataFrame with at least 'ds' and 'y' columns.
    train_ratio : float, default=0.80
        Fraction of data for training.

    Returns
    -------
    tuple of (train_df, test_df)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if len(df) < 30:
        raise ValueError(f"Need at least 30 rows for split, got {len(df)}")

    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def resample_granularity(df: pd.DataFrame, granularity: str = "Daily") -> pd.DataFrame:
    """
    Resample to a coarser granularity.

    Replicates the paper's Section 3.1 experiment of testing models at
    three different time intervals: Daily, Weekly, Monthly.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ds' (datetime) and 'y' (price) columns.
    granularity : {'Daily', 'Weekly', 'Monthly'}

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame, same column structure.
    """
    if granularity == "Daily":
        return df.copy()

    freq_map = {"Weekly": "W", "Monthly": "ME"}
    if granularity not in freq_map:
        raise ValueError(
            f"Unknown granularity: {granularity}. "
            f"Use one of: Daily, Weekly, Monthly."
        )

    return (
        df.set_index("ds")
          .resample(freq_map[granularity])
          .last()
          .dropna()
          .reset_index()
    )

def add_moving_averages(
    df: pd.DataFrame,
    sma_windows: tuple = (20, 50, 200),
    ema_windows: tuple = (20,),
) -> pd.DataFrame:
    """
    Add SMA and EMA columns for chart overlays.

    Not from the paper — added because the assignment rubric mentions
    "Technical Indicators (Optional): Moving Averages (SMA/EMA)".

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'y' column.
    sma_windows, ema_windows : tuple of int
        Window sizes for the moving averages.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added 'sma_20', 'sma_50', 'ema_20', etc.
    """
    out = df.copy()
    for w in sma_windows:
        out[f"sma_{w}"] = out["y"].rolling(window=w, min_periods=1).mean()
    for w in ema_windows:
        out[f"ema_{w}"] = out["y"].ewm(span=w, adjust=False).mean()
    return out
