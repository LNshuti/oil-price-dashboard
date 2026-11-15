---
title: Fleet Fuel Price Outlook
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: mit
---

[![CI](https://github.com/LNshuti/oil-price-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/LNshuti/oil-price-dashboard/actions/workflows/ci.yml)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LNshuti/oil-price-dashboard)

# Fleet Fuel Price Outlook

## Live Demo

Try the interactive dashboard: [HuggingFace Spaces](https://huggingface.co/spaces/LNshuti/oil-price-dashboard)

## Overview

A fuel-budgeting dashboard for fleet managers. It shows weekly retail
**diesel** and **regular gasoline** prices from the U.S. Energy
Information Administration (EIA) for every market EIA measures, and
forecasts monthly prices with uncertainty ranges.

## Features

- **ZIP-code lookup**: maps a ZIP to the most local market EIA prices for
  the selected fuel (metro area, state, PADD region or U.S.) and says
  which one it used. EIA publishes no county or ZIP-level prices.
- **Headline numbers first**: this week's price, next month's forecast
  with an 80% range, and a fuel budget for your monthly gallons.
- **Forward forecasts**: AutoARIMA, AutoETS (StatsForecast), LightGBM and
  XGBoost (MLForecast). Each is backtested on the most recent months; the
  most accurate drives the outlook, refit on all data to look ahead.
- **Markets**: every measured market side by side, with 4-week, 1-year
  and vs-U.S. changes.
- **Model check**: backtest chart and error table for every model.
- **Fresh data**: downloads EIA's weekly spreadsheets (no API key) when
  the cache is over a day old, and says plainly if a refresh fails.
- **CSV downloads** of the forecast and the weekly history.

## Setup

### Create a separate environment to isolate requirements
```bash
conda env create -f environment.yaml
```

### Install requirements

```bash
pip install -r requirements.txt
```

### Activate the environment
```bash
conda activate oil-prices
```
### Run the notebook for Exploratory data analysis
```bash
jupyter notebook
```
Navigate to the explore_oil_markets.ipynb notebook and run the cells to perform the analysis and view the visualizations.

### Run the Gradio Dashboard
```bash
python app.py
```
Navigate to [http://127.0.0.1:7860/](http://127.0.0.1:7860/) to interact with the app

### Data
- `data/processed/eia_weekly_retail.csv.gz`: cached EIA weekly prices
  (date, fuel, area, price), refreshed automatically.
- `data/reference/us_zip_codes.csv.gz`: ZIP codes with place, state,
  county and coordinates from [GeoNames](https://www.geonames.org/),
  licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Run the legacy FastAPI Dashboard
```bash
uvicorn main:app --reload
```
Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to interact with the legacy app

# References 
1. Sean Taylor. Lineapy Notebook. https://github.com/seanjtaylor/gas-price-forecast.git
