from typing import Optional 
import pandas as pd
import numpy as np 



DATE_COLUMN_CANDIDATES = ['date','timestamp','time', 'datetime', 'day']


PRICE_COLUMN_CANDIDATES = ['close','open', 'high', 'low','price', 'weighted_price']

class DataLoadError(Exception):
    """Raised when CSV cannot be loaded or validated."""
    pass

def detect_date_column(df: pd.DataFrame) -> str:
    """
    Find which column contains the dates.

    Strategy: lowercase all column names, look for any that match
    common date column names. Return the first match.
    """
    for col in df.columns:
        if col.lower() in DATE_COLUMN_CANDIDATES:
            return col
    raise DataLoadError(
         f"No date column found. Expected one of: {DATE_COLUMN_CANDIDATES}. "
        f"Got columns: {list(df.columns)}"
    )        

def detect_price_columns(df: pd.DataFrame) -> list:
    """Return list of candidate price columns found in the dataframe."""
    found = []
    for col in df.columns:
        if col.lower().strip() in PRICE_COLUMN_CANDIDATES:
            found.append(col)
    if not found:
        raise DataLoadError(
            f"No price column found. Expected one of: {PRICE_COLUMN_CANDIDATES}. "
            f"Got columns: {list(df.columns)}"
        )
    return found


def parse_dates(series:pd.Series) -> pd.Series:
    """
    Convert a date column to pandas datetime.

    Handles two cases:
    1. Unix timestamps (seconds since 1970) - common in Kaggle BTC data
    2. String dates like '2024-01-15' or '01/15/2024'
    """
    # If numeric, assume Unix timestamp
    if pd.api.types.is_numeric_dtype(series):
        # Heuristic: if value > 10^10, it's milliseconds, else seconds
        sample = series.dropna().iloc[0]
        unit = 'ms' if sample > 1e10 else 's'
        return pd.to_datetime(series, unit=unit, errors='coerce')

    # Otherwise try string parsing
    return pd.to_datetime(series, errors='coerce')
    

def load_btc_csv(
    file,
    price_column: Optional[str]= None,
) -> pd.DataFrame:
    """
    Load and validate a Bitcoin CSV file.

    Parameters
    ----------
    file : file-like object or path
        The CSV file (from Streamlit uploader or local path).
    price_column : str, optional
        Which price column to use. If None, defaults to first available
        ('close' preferred).

    Returns
    -------
    pd.DataFrame
        DataFrame with two columns:
            - 'ds' : datetime (Prophet's expected name)
            - 'y'  : float price (Prophet's expected name)
        Sorted chronologically, missing days forward-filled.

    Raises
    ------
    DataLoadError
        If the CSV is invalid, has no date column, or no price column.
    """
    # Step 1: Try to read the CSV
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise DataLoadError(f"could not read CSV file : {e}")

    if df.empty:
        raise DataLoadError("CSV file is empty.")
    
    # Step 2: Detect columns
    date_col = detect_date_column(df)
    price_cols = detect_price_columns(df)
    
    # Step 3: Choose the price column
    if price_column is None:
        for prefered in ['Close','close','CLOSE']:
            if prefered in price_cols:
                price_column = prefered
                break
        else:
            price_column = price_cols[0]
    
    if price_column not in df.columns:
        raise DataLoadError(
             f"Price column '{price_column}' not found in CSV."
             f"Available columns: {price_cols}"
             )
    
        # Step 4: Parse the dates
    df[date_col]  = parse_dates(df[date_col])
    if df[date_col].isna().all():
        raise DataLoadError(
            f"Could not parse any dates in column '{date_col}'."
            "Expected formats: '2024-01-15', '01/15/2024', or Unix timestamp."
        )        

      # Step 5: Build the clean output dataframe
    clean = pd.DataFrame({
        'ds' : df[date_col],
        'y'  : pd.to_numeric(df[price_column], errors='coerce')
    
    })
    
     
    clean = clean.dropna(subset = ['ds','y'])

    if clean.empty:
        raise DataLoadError(
            "No valid rows after cleaning. Check date and price column formats."
        )
    
        # Step 6: Sort chronologically (assignment requirement)
    clean = clean.sort_values('ds').reset_index(drop=True)


    # Step 7: Resample to daily and forward-fill missing days
    # This handles weekends (no trading) and Kaggle's minute-level data
    clean = clean.set_index('ds').resample('D').last().ffill().reset_index()

    return clean

def get_available_price_columns(file) -> list:
    """Peek at a CSV and return what price columns are available (for UI dropdown)."""
    df = pd.read_csv(file, nrows=1)
    return detect_price_columns(df)
    