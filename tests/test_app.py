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
