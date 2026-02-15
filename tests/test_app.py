import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

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


def test_create_tufte_forecast_plot_returns_figure():
    """Test that Tufte plot function returns a matplotlib figure."""
    from app import load_oil_data, run_forecasts, create_tufte_forecast_plot
    import matplotlib.pyplot as plt

    df = load_oil_data()
    results = run_forecasts(df, horizon=4)
    fig = create_tufte_forecast_plot(df, results['forecasts'])

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


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


def test_gradio_app_builds():
    """Test that the Gradio app can be instantiated."""
    from app import create_app
    import gradio as gr

    app = create_app()

    assert isinstance(app, gr.Blocks)
