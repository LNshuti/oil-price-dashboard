[![CI](https://github.com/LNshuti/oil-price-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/LNshuti/oil-price-dashboard/actions/workflows/ci.yml)

# Oil Price Dashboard Project

## Overview

The Oil Price Dashboard Project is designed to understand the US oil markets by forecasting prices and production quantities. This project uses data from the U.S. Energy Information Administration, focusing on the wholesale gasoline prices. The core of the analysis is performed in a Jupyter Notebook, `explore_oil_markets.ipynb`, which includes data loading, cleaning, transformation, forecasting and visualization.

```mermaid
graph TD;
    subgraph Frontend
        react["React.js + Redux"] --> materialui["Material-UI"]
        materialui --> chartjs["Chart.js"]
        react --> s3frontend["AWS S3 (Static Hosting)"]
    end

    subgraph Backend
        fastapi["FastAPI"] --> redis["Redis (Caching)"]
        fastapi --> rds["RDS PostgreSQL (Database)"]
        fastapi --> s3data["AWS S3 (Data Storage)"]
        fastapi --> glue["AWS Glue (ETL)"]
        fastapi --> dynamodb["DynamoDB (User Feedback)"]
        glue --> ml["Machine Learning Processing"]
        ml --> fastapi
    end

    subgraph Machine_Learning
        sklearn["Scikit-learn"] --> tensorflow["TensorFlow"]
        sklearn --> pytorch["PyTorch"]
        tensorflow --> training["Model Training"]
        pytorch --> training
        training --> infer["Inference Models"]
        infer --> fastapi
    end

    subgraph Deployment
        docker["Docker"] --> kubernetes["Kubernetes (AWS EKS)"]
        kubernetes --> elasticbeanstalk["AWS Elastic Beanstalk"]
        elasticbeanstalk --> fastapi
    end

    subgraph CI_CD
        github["GitHub Actions"] --> docker
        docker --> ci["CI/CD Pipeline"]
        ci --> deployment["Deployment Process"]
        deployment --> kubernetes
    end

    style react fill:#f9f,stroke:#333,stroke-width:2px
    style fastapi fill:#bbf,stroke:#333,stroke-width:2px
    style sklearn fill:#fbb,stroke:#333,stroke-width:2px
    style docker fill:#ddf,stroke:#333,stroke-width:2px
```

## Features

- **Data Analysis**: Jupyter [Notebook]("https://github.com/LNshuti/oil-price-dashboard/blob/main/'explore_oil_markets.ipynb'") contains detailed steps for data cleaning, transformation, and preliminary analysis. 

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

### Run the interactive Dashboard
```bash
uvicorn main:app --reload
```
#### Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to interact with the app

# References 
1. Sean Taylor. Lineapy Notebook. https://github.com/seanjtaylor/gas-price-forecast.git
