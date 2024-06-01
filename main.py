# main.py (complete)

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
import re
from statsforecast.models import AutoARIMA
import numpy as np
import altair as alt

app = FastAPI()

# Constants
CI = 95  # Confidence Interval
H = 14  # Forecast horizon in days

# Model for shipping cost request
class ShippingCostRequest(BaseModel):
    origin: str
    destination: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Data fetching and forecasting function
def fetch_oil_price_data():
    url = "https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls"
    response = requests.get(url)
    df = pd.read_excel(
        response.content,
        sheet_name="Data 12",
        index_col=0,
        skiprows=2,
        parse_dates=["Date"]
    ).rename(
        columns=lambda c: re.sub(
            "\(PADD 1[A-C]\)",
            "",
            c.replace("Weekly ", "").replace(
                " All Grades All Formulations Retail Gasoline Prices  (Dollars per Gallon)",
                "",
            ),
        ).strip()
    )
    return df

def forecast_prices(data):
    forecast = pd.DataFrame()
    for column in data.columns:
        series = data[column].dropna()
        model = AutoARIMA()
        model.fit(series)
        raw_forecast = model.predict(h=H, level=(CI,))
        raw_forecast_exp = {key: np.exp(value) for key, value in raw_forecast.items()}
        forecast[column] = pd.DataFrame(raw_forecast_exp)['mean']
    return forecast

@app.get("/forecast")
def get_forecast():
    data = fetch_oil_price_data()
    forecast = forecast_prices(data)
    return forecast.to_dict()

# Dummy shipping cost function for demonstration purposes
def calculate_shipping_cost(origin, destination):
    shipping_costs = {
        "Mississippi": {"Liberia": 2000, "Dar Es Salaam": 3000},
        "Guyana": {"Liberia": 2500, "Dar Es Salaam": 3500},
    }
    return shipping_costs.get(origin, {}).get(destination, "Route not available")

@app.post("/shipping_cost")
def get_shipping_cost(request: ShippingCostRequest):
    cost = calculate_shipping_cost(request.origin, request.destination)
    return {"origin": request.origin, "destination": request.destination, "cost": cost}
