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

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


def prepare_data_for_nixtla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data in Nixtla's expected format.

    Nixtla expects: unique_id, ds, y
    """
    nixtla_df = df.reset_index()
    nixtla_df.columns = ['ds', 'y']
    nixtla_df['unique_id'] = 'oil_price'
    # Drop NaN values and aggregate duplicates by taking the mean
    nixtla_df = nixtla_df.dropna(subset=['y'])
    nixtla_df = nixtla_df.groupby('ds', as_index=False).agg({'y': 'mean', 'unique_id': 'first'})
    nixtla_df = nixtla_df.sort_values('ds').reset_index(drop=True)
    return nixtla_df[['unique_id', 'ds', 'y']]


def run_forecasts(df: pd.DataFrame, horizon: int = 12) -> dict:
    """
    Run multiple forecasting models and return predictions with metrics.

    Args:
        df: DataFrame with datetime index and 'price' column.
        horizon: Number of periods to forecast.

    Returns:
        Dictionary with 'forecasts' DataFrame and 'metrics' DataFrame.
    """
    # Prepare data
    nixtla_df = prepare_data_for_nixtla(df.rename(columns={'price': 'y'}) if 'price' in df.columns else df)

    # Split for validation
    train_size = len(nixtla_df) - horizon
    train_df = nixtla_df.iloc[:train_size]
    test_df = nixtla_df.iloc[train_size:]

    results = {}
    metrics_list = []

    # Statistical models with StatsForecast
    sf = StatsForecast(
        models=[
            AutoARIMA(season_length=12),
            AutoETS(season_length=12),
        ],
        freq='MS',  # Month start
        n_jobs=-1
    )

    sf.fit(train_df)
    sf_preds = sf.predict(h=horizon, level=[80, 95])

    # ML models with MLForecast
    mlf = MLForecast(
        models={
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                verbose=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                verbosity=0
            ),
        },
        freq='MS',
        lags=[1, 2, 3, 6, 12],
        lag_transforms={
            1: [ExpandingMean()],
            3: [RollingMean(window_size=3)],
        },
    )

    mlf.fit(train_df)
    mlf_preds = mlf.predict(h=horizon)

    # Combine forecasts
    all_forecasts = sf_preds.merge(mlf_preds, on=['unique_id', 'ds'], how='outer')

    # Calculate metrics
    actual = test_df['y'].values

    for model_name in ['AutoARIMA', 'AutoETS', 'LightGBM', 'XGBoost']:
        if model_name in all_forecasts.columns:
            preds = all_forecasts[model_name].values[:len(actual)]
            mae = mean_absolute_error(actual, preds)
            rmse = np.sqrt(mean_squared_error(actual, preds))
            mape = np.mean(np.abs((actual - preds) / actual)) * 100

            metrics_list.append({
                'Model': model_name,
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'MAPE': round(mape, 2)
            })

    metrics_df = pd.DataFrame(metrics_list)

    return {
        'forecasts': all_forecasts,
        'metrics': metrics_df,
        'train_df': train_df,
        'test_df': test_df,
    }
