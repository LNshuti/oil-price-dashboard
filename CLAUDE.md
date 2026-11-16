# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fleet fuel price outlook for **fleet managers** (the target user):

- **Gradio dashboard** (`app.py` + `dashboard/` package): weekly EIA retail
  diesel and regular gasoline prices for every market EIA measures, ZIP
  lookup to the nearest measured market, monthly forecasts and a fuel
  budget. Deployed to HuggingFace Spaces (`app_file: app.py`).
- **Legacy, unrelated to the dashboard**: `backend/server/` (Django ML
  API scaffold whose only model is an income classifier).

Data source: EIA weekly retail spreadsheets (`psw18vwall.xls` diesel
"Data 1", `pswrgvwall.xls` regular gasoline "Data 3"); no API key needed.
The EIA API's DEMO_KEY rate-limits after a handful of calls, so don't use
it for backfills.

## Common Commands

### Environment Setup
```bash
conda env create -f environment.yaml
conda activate oil-prices
pip install -r requirements.txt
```

### Run Dashboard
```bash
python app.py
# Access at http://127.0.0.1:7860/
```

### Run Django Backend
```bash
cd backend/server
python manage.py runserver
```

### Testing
```bash
pytest --cov=dashboard --cov=app tests    # all tests (~15s), as CI runs them
pytest tests/test_app.py                  # dashboard tests only
pytest -k "test_name"                     # run specific test by name
```
CI (`.github/workflows/ci.yml`) runs flake8 and pytest on Python 3.10
(HuggingFace Spaces default) and 3.12. Gradio is pinned `>=6.31,<7` in
requirements.txt and `sdk_version` in README.md; keep them in sync.

### Linting
```bash
flake8 --max-line-length=79 --max-complexity=10
```

## Architecture

### Dashboard (`dashboard/`)
- `data.py`: downloads both EIA spreadsheets, caches long-format prices in
  `data/processed/eia_weekly_retail.csv.gz` (refreshed when >24h old; a
  failed refresh keeps the cache and is reported in the UI). `monthly()`
  drops the month still in progress.
- `location.py`: market catalog (EIA area codes, parent chain city ->
  state -> PADD sub-region -> PADD -> U.S.), state->PADD map, and ZIP
  resolution using `data/reference/us_zip_codes.csv.gz` (GeoNames, CC BY).
  Diesel covers only 11 markets (no cities/most states), so resolution
  always falls back through the chain to a market the fuel has.
- `forecast.py`: backtest all four models on the last `horizon` months,
  recommend the lowest-MAPE model, refit on full history (last 20 years)
  and forecast forward with 80/95% intervals (conformal for ML models).
  Outlooks are memoized by (fuel, area, horizon, data as-of date).
- `charts.py` (Plotly) and `views.py` (HTML tiles/tables). Each model keeps
  one color everywhere (validated categorical palette).
- `app.py`: one handler per control that returns every output at once;
  don't chain `.then()` steps between controls, it caused stale renders.
  Gradio 6: `css`/`theme` go to `launch()`, not `Blocks()`.

### Django ML Backend Structure (legacy)
- `backend/server/apps/endpoints/`: REST API views, serializers, models for algorithm registry
- `backend/server/apps/ml/`: ML model implementations and registry
- Serverless deployment config in `serverless.yaml` for AWS Lambda
