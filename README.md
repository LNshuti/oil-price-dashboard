---
title: Oil Price Dashboard
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
pinned: false
license: mit
---

[![CI](https://github.com/LNshuti/oil-price-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/LNshuti/oil-price-dashboard/actions/workflows/ci.yml)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LNshuti/oil-price-dashboard)

# Oil Price Dashboard Project

## Live Demo

Try the interactive dashboard: [HuggingFace Spaces](https://huggingface.co/spaces/LNshuti/oil-price-dashboard)

## Overview

The Oil Price Dashboard Project is designed to understand the US oil markets by forecasting prices and production quantities. This project uses data from the U.S. Energy Information Administration, focusing on the wholesale gasoline prices.

## Features

- **Interactive Gradio Dashboard**: Enterprise-grade forecasting dashboard with state-of-the-art ML models
- **ML Models**: AutoARIMA, AutoETS (StatsForecast), LightGBM, XGBoost (MLForecast)
- **Tufte-styled Visualizations**: Minimalist charts following Edward Tufte's principles
- **NEJM-formatted Tables**: Professional academic-style data tables
- **Data Analysis**: Jupyter Notebook for exploratory analysis

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

### Run the legacy FastAPI Dashboard
```bash
uvicorn main:app --reload
```
Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to interact with the legacy app

# References 
1. Sean Taylor. Lineapy Notebook. https://github.com/seanjtaylor/gas-price-forecast.git
