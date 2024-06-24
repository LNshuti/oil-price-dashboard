[![Build Status](https://app.travis-ci.com/LNshuti/oil-price-dashboard.svg?branch=main)](https://app.travis-ci.com/LNshuti/oil-price-dashboard)

# Oil Price Dashboard Project

## Overview

The Oil Price Dashboard Project is designed to undersand the US oil markets by forecasting prices and production quantities through an interactive dashboard. This project utilizes data from various sources, focusing on the wholesale gasoline prices in different regions. The core of the analysis is performed in a Jupyter Notebook, `explore_oil_markets.ipynb`, which includes data loading, cleaning, transformation, and visualization steps to understand the trends and patterns in oil prices over time.

## Features

- **Data Analysis**: Jupyter Notebook contains detailed steps for data cleaning, transformation, and preliminary analysis. It explores various aspects of the oil price data, such as trends over time, comparisons between different regions, and more.

- **Interactive Dashboard**: Integrates the analysis and visualizations into an interactive dashboard using FastAPI.

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


# References 
1. Sean Taylor. Lineapy Notebook. https://github.com/seanjtaylor/gas-price-forecast.git
