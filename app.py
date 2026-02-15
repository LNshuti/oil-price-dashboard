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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


def setup_tufte_style():
    """Configure matplotlib for Tufte-inspired minimalist style."""
    plt.rcParams.update({
        # Remove chartjunk
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # Minimal grid
        'axes.grid': False,

        # Muted colors
        'axes.prop_cycle': plt.cycler(color=[
            '#4a4a4a',  # Dark gray
            '#e63946',  # Accent red
            '#457b9d',  # Steel blue
            '#2a9d8f',  # Teal
        ]),

        # Typography
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,

        # Clean ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,

        # Figure
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.dpi': 100,
    })


def create_tufte_forecast_plot(
    historical_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    title: str = "Oil Price Forecast"
) -> plt.Figure:
    """
    Create a Tufte-styled forecast plot with confidence intervals.

    Args:
        historical_df: Historical price data.
        forecasts_df: Forecast results from run_forecasts.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    setup_tufte_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical data
    hist_dates = historical_df.index
    hist_values = historical_df['price'] if 'price' in historical_df.columns else historical_df.iloc[:, 0]

    ax.plot(hist_dates, hist_values, color='#4a4a4a', linewidth=1.5, label='Historical')

    # Forecast data
    forecast_dates = pd.to_datetime(forecasts_df['ds'])

    # Plot each model's forecast
    model_colors = {
        'AutoARIMA': '#e63946',
        'AutoETS': '#457b9d',
        'LightGBM': '#2a9d8f',
        'XGBoost': '#f4a261',
    }

    for model_name, color in model_colors.items():
        if model_name in forecasts_df.columns:
            ax.plot(
                forecast_dates,
                forecasts_df[model_name],
                color=color,
                linewidth=1.5,
                linestyle='--',
                label=model_name
            )

            # Confidence intervals (if available)
            lo_col = f'{model_name}-lo-80'
            hi_col = f'{model_name}-hi-80'
            if lo_col in forecasts_df.columns and hi_col in forecasts_df.columns:
                ax.fill_between(
                    forecast_dates,
                    forecasts_df[lo_col],
                    forecasts_df[hi_col],
                    color=color,
                    alpha=0.15,
                )

    # Minimal decoration
    ax.set_title(title, fontweight='normal', loc='left', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('Price ($/gallon)', fontsize=10)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # Legend - direct labeling style (outside plot, minimal)
    ax.legend(
        loc='upper left',
        frameon=False,
        fontsize=9,
    )

    # Thin axis lines
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color('#888888')

    plt.tight_layout()

    return fig


def create_model_comparison_plot(metrics_df: pd.DataFrame) -> plt.Figure:
    """Create a Tufte-styled bar chart comparing model metrics."""
    setup_tufte_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = ['MAE', 'RMSE', 'MAPE']
    colors = ['#457b9d', '#e63946', '#2a9d8f']

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(
            metrics_df['Model'],
            metrics_df[metric],
            color=color,
            width=0.6,
            edgecolor='none'
        )

        # Direct labels on bars
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02 * max(metrics_df[metric]),
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_title(metric, fontweight='normal', loc='left')
        ax.set_ylabel('')
        ax.set_xlabel('')

        # Remove top spine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


NEJM_CSS = """
<style>
.nejm-table {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    font-size: 14px;
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}

.nejm-table thead {
    border-top: 2px solid #333;
    border-bottom: 1px solid #333;
}

.nejm-table th {
    text-align: left;
    padding: 12px 16px;
    font-weight: 600;
    color: #333;
}

.nejm-table th.numeric {
    text-align: right;
}

.nejm-table tbody tr {
    border-bottom: 1px solid #e0e0e0;
}

.nejm-table tbody tr:nth-child(even) {
    background-color: #f9f9f9;
}

.nejm-table tbody tr:last-child {
    border-bottom: 2px solid #333;
}

.nejm-table td {
    padding: 10px 16px;
    color: #444;
}

.nejm-table td.numeric {
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.nejm-table td.model-name {
    font-weight: 500;
}
</style>
"""


def format_nejm_table(df: pd.DataFrame, caption: str = None) -> str:
    """
    Format a DataFrame as an NEJM-styled HTML table.

    NEJM style:
    - Horizontal rules only (top, header-bottom, table-bottom)
    - No vertical lines
    - Right-aligned numbers, left-aligned text
    - Subtle alternating row shading

    Args:
        df: DataFrame to format.
        caption: Optional table caption.

    Returns:
        HTML string with embedded CSS.
    """
    html_parts = [NEJM_CSS, '<table class="nejm-table">']

    if caption:
        html_parts.append(f'<caption style="text-align: left; font-weight: 600; margin-bottom: 8px;">{caption}</caption>')

    # Header
    html_parts.append('<thead><tr>')
    for col in df.columns:
        # Detect numeric columns for alignment
        is_numeric = df[col].dtype in ['float64', 'int64', 'float32', 'int32']
        align_class = 'numeric' if is_numeric else ''
        html_parts.append(f'<th class="{align_class}">{col}</th>')
    html_parts.append('</tr></thead>')

    # Body
    html_parts.append('<tbody>')
    for _, row in df.iterrows():
        html_parts.append('<tr>')
        for i, (col, val) in enumerate(row.items()):
            is_numeric = isinstance(val, (int, float))
            if is_numeric and not pd.isna(val):
                cell_class = 'numeric'
                formatted_val = f'{val:.4f}' if isinstance(val, float) else str(val)
            else:
                cell_class = 'model-name' if i == 0 else ''
                formatted_val = str(val)
            html_parts.append(f'<td class="{cell_class}">{formatted_val}</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody></table>')

    return '\n'.join(html_parts)


def create_forecast_table(forecasts_df: pd.DataFrame) -> str:
    """Create NEJM-styled table of forecast values."""
    # Prepare display dataframe
    display_df = forecasts_df.copy()
    display_df['Date'] = pd.to_datetime(display_df['ds']).dt.strftime('%Y-%m')

    # Select key columns
    model_cols = ['AutoARIMA', 'AutoETS', 'LightGBM', 'XGBoost']
    available_cols = ['Date'] + [c for c in model_cols if c in display_df.columns]
    display_df = display_df[available_cols]

    return format_nejm_table(display_df, caption="Forecasted Values by Model")
