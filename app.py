"""
Oil Price Forecasting Dashboard
Enterprise-grade Gradio app with state-of-the-art ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
from io import BytesIO

# Constants
DATA_DIR = Path("data/processed")
CACHE_FILE = DATA_DIR / "oil_prices_cache.csv"
CACHE_DURATION_HOURS = 24
EIA_URL = "https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls"


def load_oil_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load oil price data from cache or fetch from EIA.

    Args:
        force_refresh: If True, bypass cache and fetch fresh data.

    Returns:
        DataFrame with datetime index and price values.
    """
    # Check for existing processed data first
    monthly_file = DATA_DIR / "louisiana_tot_gasoline_wholesale_monthly.csv"

    if monthly_file.exists() and not force_refresh:
        df = pd.read_csv(monthly_file)
        # Extract date from the 'data' column which contains ['YYYYMM', value] format
        # The 'date' column in the file is inconsistent, so we parse from 'data'
        if 'data' in df.columns:
            # Extract YYYYMM from data column like "['202203', 3.09]"
            df['date'] = df['data'].str.extract(r"'(\d{6})'")[0]
        # Drop rows where date couldn't be extracted
        df = df.dropna(subset=['date'])
        df['date'] = pd.to_datetime(df['date'].astype(str) + '01', format='%Y%m%d')
        df = df.set_index('date')
        df = df.sort_index()
        df = df[['value']].rename(columns={'value': 'price'})
        return df

    # Fallback to cache
    if CACHE_FILE.exists() and not force_refresh:
        cache_time = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        if datetime.now() - cache_time < timedelta(hours=CACHE_DURATION_HOURS):
            df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            return df

    # Fetch fresh data
    df = fetch_eia_data()

    # Cache the data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE)

    return df


def fetch_eia_data() -> pd.DataFrame:
    """Fetch latest oil price data from EIA."""
    try:
        response = requests.get(EIA_URL, timeout=30)
        response.raise_for_status()

        df = pd.read_excel(
            BytesIO(response.content),
            sheet_name="Data 12",
            skiprows=2
        )

        # Clean and process
        df = df.iloc[:, :2]  # First two columns: date and price
        df.columns = ['date', 'price']
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna()
        df = df.set_index('date')
        df = df.sort_index()

        return df
    except Exception as e:
        # Fall back to local data if fetch fails
        monthly_file = DATA_DIR / "louisiana_tot_gasoline_wholesale_monthly.csv"
        if monthly_file.exists():
            df = pd.read_csv(monthly_file)
            # Extract date from the 'data' column
            if 'data' in df.columns:
                df['date'] = df['data'].str.extract(r"'(\d{6})'")[0]
            # Drop rows where date couldn't be extracted
            df = df.dropna(subset=['date'])
            df['date'] = pd.to_datetime(df['date'].astype(str) + '01', format='%Y%m%d')
            df = df.set_index('date')
            df = df.sort_index()
            df = df[['value']].rename(columns={'value': 'price'})
            return df
        raise RuntimeError(f"Failed to fetch data and no local cache: {e}")
