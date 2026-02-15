# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Oil price forecasting dashboard with two main components:
- **FastAPI Dashboard** (`scr/main.py`): Interactive visualization of oil price forecasts using Random Forest and XGBoost models
- **Django ML API** (`backend/server/`): REST API for ML model serving with model registry and A/B testing support

Data source: U.S. Energy Information Administration (EIA) - Louisiana wholesale gasoline prices.

## Common Commands

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate oil-prices
pip install -r requirements.txt
```

### Run Dashboard
```bash
cd scr
uvicorn main:app --reload
# Access at http://127.0.0.1:8000/
```

### Run Django Backend
```bash
cd backend/server
python manage.py runserver
```

### Testing
```bash
pytest --cov=src tests                    # Run all tests with coverage
pytest tests/data/test_download_oil_prices.py  # Run single test file
pytest -k "test_name"                     # Run specific test by name
```

### Linting
```bash
flake8 --max-line-length=79 --max-complexity=10
```

## Architecture

### Data Flow
1. `scr/data/download_oil_prices.py` fetches EIA data from `pswrgvwall.xls`
2. Processed data stored in `data/processed/` as CSV
3. `scr/main.py` loads data, engineers features (12 lag features + rolling stats), trains models at startup
4. Dashboard renders Plotly charts comparing RF vs XGBoost predictions

### Django ML Backend Structure
- `backend/server/apps/endpoints/`: REST API views, serializers, models for algorithm registry
- `backend/server/apps/ml/`: ML model implementations and registry
- Supports algorithm states: testing, staging, production, ab_testing
- Serverless deployment config in `serverless.yaml` for AWS Lambda

### Key Data Transformations
The dashboard creates these features from raw price data:
- `lag_1` through `lag_12`: Previous 12 months of prices
- `rolling_mean_3`, `rolling_std_3`: 3-month rolling statistics

## Testing Notes

- `tests/visualization/test_plots.py` uses pytest-mpl for image comparison
- Data tests mock external EIA API calls
- CI runs via GitHub Actions with Codecov integration
