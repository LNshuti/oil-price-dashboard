# Oil Price Dashboard Modernization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build enterprise-grade Gradio app with Nixtla ML stack, Tufte charts, NEJM tables, deployed to HuggingFace Spaces.

**Architecture:** Single-file Gradio app using StatsForecast/MLForecast for forecasting, Matplotlib with custom Tufte styling, and Gradio Blocks for layout. Data cached locally with on-demand refresh from EIA.

**Tech Stack:** Gradio, StatsForecast, MLForecast, LightGBM, XGBoost, Matplotlib, Pandas

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Add Nixtla stack and Gradio dependencies:

```
altair
matplotlib
numpy
pandas
seaborn
statsmodels
scikit-learn
fastapi
uvicorn
plotly
xgboost
gradio>=4.0.0
statsforecast
mlforecast
lightgbm
requests
```

**Step 2: Verify dependencies install**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "deps: add gradio and nixtla forecasting stack"
```

---

## Task 2: Create Data Module

**Files:**
- Create: `app.py` (data section)

**Step 1: Write failing test for data loading**

Create `tests/test_app.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


def test_load_oil_data_returns_dataframe():
    """Test that load_oil_data returns a DataFrame with expected columns."""
    # Import after creating the module
    from app import load_oil_data

    df = load_oil_data()

    assert isinstance(df, pd.DataFrame)
    assert 'date' in df.columns or df.index.name == 'date'
    assert 'value' in df.columns or 'price' in df.columns
    assert len(df) > 0


def test_load_oil_data_has_datetime_index():
    """Test that the data has a proper datetime index."""
    from app import load_oil_data

    df = load_oil_data()

    if df.index.name == 'date':
        assert pd.api.types.is_datetime64_any_dtype(df.index)
    else:
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL with "No module named 'app'" or "cannot import name 'load_oil_data'"

**Step 3: Write minimal data loading implementation**

Create `app.py` with data loading:

```python
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
        df['date'] = pd.to_datetime(df['date'].astype(str) + '01', format='%Y%m%d')
        df = df.set_index('date')
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
            df['date'] = pd.to_datetime(df['date'].astype(str) + '01', format='%Y%m%d')
            df = df.set_index('date')
            df = df[['value']].rename(columns={'value': 'price'})
            return df
        raise RuntimeError(f"Failed to fetch data and no local cache: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add data loading module with EIA integration"
```

---

## Task 3: Create Forecasting Module

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Step 1: Write failing test for forecasting**

Add to `tests/test_app.py`:

```python
def test_run_forecasts_returns_results():
    """Test that forecasting returns predictions for all models."""
    from app import load_oil_data, run_forecasts

    df = load_oil_data()
    results = run_forecasts(df, horizon=4)

    assert isinstance(results, dict)
    assert 'forecasts' in results
    assert 'metrics' in results
    assert len(results['forecasts']) > 0


def test_run_forecasts_includes_confidence_intervals():
    """Test that forecasts include confidence intervals."""
    from app import load_oil_data, run_forecasts

    df = load_oil_data()
    results = run_forecasts(df, horizon=4)

    forecasts_df = results['forecasts']
    # Check for lo/hi columns (confidence intervals)
    ci_cols = [c for c in forecasts_df.columns if 'lo' in c.lower() or 'hi' in c.lower()]
    assert len(ci_cols) > 0, "No confidence interval columns found"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_run_forecasts_returns_results -v`
Expected: FAIL with "cannot import name 'run_forecasts'"

**Step 3: Write forecasting implementation**

Add to `app.py`:

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_data_for_nixtla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data in Nixtla's expected format.

    Nixtla expects: unique_id, ds, y
    """
    nixtla_df = df.reset_index()
    nixtla_df.columns = ['ds', 'y']
    nixtla_df['unique_id'] = 'oil_price'
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py::test_run_forecasts_returns_results tests/test_app.py::test_run_forecasts_includes_confidence_intervals -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add Nixtla forecasting with StatsForecast and MLForecast"
```

---

## Task 4: Create Tufte Visualization Theme

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Step 1: Write failing test for Tufte plot**

Add to `tests/test_app.py`:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing


def test_create_tufte_forecast_plot_returns_figure():
    """Test that Tufte plot function returns a matplotlib figure."""
    from app import load_oil_data, run_forecasts, create_tufte_forecast_plot
    import matplotlib.pyplot as plt

    df = load_oil_data()
    results = run_forecasts(df, horizon=4)
    fig = create_tufte_forecast_plot(df, results['forecasts'])

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_create_tufte_forecast_plot_returns_figure -v`
Expected: FAIL with "cannot import name 'create_tufte_forecast_plot'"

**Step 3: Write Tufte visualization implementation**

Add to `app.py`:

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py::test_create_tufte_forecast_plot_returns_figure -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add Tufte-styled visualization functions"
```

---

## Task 5: Create NEJM Table Styling

**Files:**
- Modify: `app.py`

**Step 1: Write failing test for NEJM table**

Add to `tests/test_app.py`:

```python
def test_format_nejm_table_returns_styled_html():
    """Test that NEJM formatter returns HTML string."""
    from app import format_nejm_table
    import pandas as pd

    df = pd.DataFrame({
        'Model': ['A', 'B'],
        'MAE': [0.1, 0.2],
        'RMSE': [0.15, 0.25]
    })

    html = format_nejm_table(df)

    assert isinstance(html, str)
    assert '<table' in html or '<style' in html
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_format_nejm_table_returns_styled_html -v`
Expected: FAIL with "cannot import name 'format_nejm_table'"

**Step 3: Write NEJM table implementation**

Add to `app.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py::test_format_nejm_table_returns_styled_html -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add NEJM-styled table formatting"
```

---

## Task 6: Create Gradio Interface

**Files:**
- Modify: `app.py`

**Step 1: Write failing test for Gradio app**

Add to `tests/test_app.py`:

```python
def test_gradio_app_builds():
    """Test that the Gradio app can be instantiated."""
    from app import create_app
    import gradio as gr

    app = create_app()

    assert isinstance(app, gr.Blocks)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py::test_gradio_app_builds -v`
Expected: FAIL with "cannot import name 'create_app'"

**Step 3: Write Gradio interface implementation**

Add to `app.py`:

```python
import gradio as gr
from io import BytesIO
import base64


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 for display."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')


def run_dashboard(horizon: int, refresh_data: bool) -> tuple:
    """
    Main dashboard callback function.

    Args:
        horizon: Forecast horizon in periods.
        refresh_data: Whether to force data refresh.

    Returns:
        Tuple of (forecast_plot, metrics_html, forecast_table_html, comparison_plot)
    """
    # Load data
    df = load_oil_data(force_refresh=refresh_data)

    # Run forecasts
    results = run_forecasts(df, horizon=horizon)

    # Create visualizations
    forecast_fig = create_tufte_forecast_plot(df, results['forecasts'])
    comparison_fig = create_model_comparison_plot(results['metrics'])

    # Create tables
    metrics_html = format_nejm_table(results['metrics'], caption="Model Performance Metrics")
    forecast_html = create_forecast_table(results['forecasts'])

    return forecast_fig, metrics_html, forecast_html, comparison_fig


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    # Custom CSS for the app
    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .app-title {
        font-size: 24px;
        font-weight: 600;
        color: #333;
        margin-bottom: 8px;
    }
    .app-subtitle {
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(css=custom_css, title="Oil Price Forecast Dashboard") as app:

        # Header
        gr.HTML("""
            <div class="app-title">Oil Price Forecasting Dashboard</div>
            <div class="app-subtitle">
                State-of-the-art forecasting with AutoARIMA, AutoETS, LightGBM, and XGBoost
            </div>
        """)

        # Controls
        with gr.Row():
            horizon_dropdown = gr.Dropdown(
                choices=[4, 8, 12, 26, 52],
                value=12,
                label="Forecast Horizon (periods)",
            )
            refresh_checkbox = gr.Checkbox(
                label="Refresh Data",
                value=False,
            )
            run_button = gr.Button("Run Forecast", variant="primary")

        # Tabs
        with gr.Tabs():
            with gr.TabItem("Forecast"):
                forecast_plot = gr.Plot(label="Price Forecast")
                forecast_table = gr.HTML(label="Forecast Values")

            with gr.TabItem("Model Comparison"):
                comparison_plot = gr.Plot(label="Model Metrics Comparison")
                metrics_table = gr.HTML(label="Performance Metrics")

            with gr.TabItem("Data Explorer"):
                data_preview = gr.DataFrame(
                    label="Historical Data",
                    interactive=False,
                )
                download_btn = gr.Button("Download CSV")
                download_file = gr.File(label="Download", visible=False)

            with gr.TabItem("About"):
                gr.Markdown("""
                ## Methodology

                This dashboard uses state-of-the-art time series forecasting methods:

                **Statistical Models (StatsForecast):**
                - **AutoARIMA**: Automatic ARIMA model selection with seasonal components
                - **AutoETS**: Automatic exponential smoothing state space model

                **Machine Learning Models (MLForecast):**
                - **LightGBM**: Gradient boosting with lag features and rolling statistics
                - **XGBoost**: Alternative gradient boosting implementation

                ## Data Source

                U.S. Energy Information Administration (EIA) - Weekly Petroleum Status Report

                ## Confidence Intervals

                Statistical models provide 80% and 95% prediction intervals.

                ---

                *Built with Gradio, Nixtla, and Matplotlib*
                """)

        # Event handlers
        def on_run_forecast(horizon, refresh):
            forecast_fig, metrics_html, forecast_html, comparison_fig = run_dashboard(
                horizon=int(horizon),
                refresh_data=refresh
            )
            return forecast_fig, metrics_html, forecast_html, comparison_fig

        def load_data_preview():
            df = load_oil_data()
            preview = df.tail(50).reset_index()
            preview.columns = ['Date', 'Price']
            return preview

        def download_data():
            df = load_oil_data()
            csv_path = "/tmp/oil_prices_export.csv"
            df.to_csv(csv_path)
            return csv_path

        run_button.click(
            fn=on_run_forecast,
            inputs=[horizon_dropdown, refresh_checkbox],
            outputs=[forecast_plot, metrics_table, forecast_table, comparison_plot]
        )

        app.load(
            fn=load_data_preview,
            outputs=[data_preview]
        )

        download_btn.click(
            fn=download_data,
            outputs=[download_file]
        )

    return app


# Main entry point
if __name__ == "__main__":
    app = create_app()
    app.launch()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py::test_gradio_app_builds -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add enterprise Gradio dashboard interface"
```

---

## Task 7: Run All Tests

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Run existing tests to check for regressions**

Run: `pytest tests/test_environment.py tests/data/ tests/visualization/ -v --tb=short`
Expected: All tests PASS (or skip gracefully)

**Step 3: Commit any fixes if needed**

---

## Task 8: Test Gradio App Locally

**Step 1: Launch the app**

Run: `python app.py`
Expected: App launches at http://127.0.0.1:7860

**Step 2: Manual verification checklist**

- [ ] App loads without errors
- [ ] "Run Forecast" button triggers computation
- [ ] Forecast plot displays with Tufte styling
- [ ] Tables display with NEJM formatting
- [ ] Model Comparison tab shows metrics
- [ ] Data Explorer shows historical data
- [ ] Download CSV works

**Step 3: Fix any issues found**

---

## Task 9: Update README for HuggingFace

**Files:**
- Modify: `README.md`

**Step 1: Add HuggingFace Spaces section**

Add to top of `README.md`:

```markdown
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/oil-price-dashboard)

## Live Demo

Try the interactive dashboard: [HuggingFace Spaces](https://huggingface.co/spaces/YOUR_USERNAME/oil-price-dashboard)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add HuggingFace Spaces badge and link"
```

---

## Task 10: Final Commit and Create PR

**Step 1: Verify all changes**

Run: `git status && git log --oneline -10`

**Step 2: Push branch**

Run: `git push -u origin feat-modernize`

**Step 3: Create PR**

Run:
```bash
gh pr create --title "Modernize dashboard with Nixtla ML stack and Gradio" --body "$(cat <<'EOF'
## Summary
- Add enterprise-grade Gradio dashboard for oil price forecasting
- Implement state-of-the-art ML models via Nixtla stack (StatsForecast + MLForecast)
- Add Tufte-styled visualizations (minimal chartjunk, high data-ink ratio)
- Add NEJM-formatted tables (professional academic style)
- Ready for HuggingFace Spaces deployment

## Models Included
- AutoARIMA (statistical baseline)
- AutoETS (exponential smoothing)
- LightGBM (gradient boosting)
- XGBoost (gradient boosting)

## Features
- Configurable forecast horizon (4-52 periods)
- On-demand data refresh
- Model comparison metrics (MAE, RMSE, MAPE)
- Confidence intervals (80%, 95%)
- CSV export

## Test plan
- [x] Unit tests for data loading
- [x] Unit tests for forecasting pipeline
- [x] Unit tests for visualization functions
- [x] Manual testing of Gradio interface
- [x] Existing tests pass (no regressions)

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

Expected: PR created successfully, URL returned

---

## Post-Implementation: HuggingFace Deployment

After PR is merged, deploy to HuggingFace Spaces:

1. Create new Space at huggingface.co/new-space
2. Select "Gradio" as SDK
3. Upload `app.py` and `requirements.txt`
4. Space auto-builds and deploys

Alternatively, connect GitHub repo for auto-sync.
