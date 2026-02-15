from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

app = FastAPI()

# Load the dataset (Gulf Coast retail gasoline prices - includes Louisiana/PADD 3)
file_path = '../data/processed/gulf_coast_gasoline_monthly.csv'
data = pd.read_csv(file_path)
# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'].astype(str) + '01', format='%Y%m%d')

# Set the date column as index
data.set_index('date', inplace=True)
data = data[['value']]  # Keep only the necessary 'value' column

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Gasoline Retail Price')
plt.title('Gulf Coast Gasoline Retail Monthly Prices (1993-2026)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Feature engineering
for lag in range(1, 13):
    data[f'lag_{lag}'] = data['value'].shift(lag)
data['rolling_mean_3'] = data['value'].rolling(window=3).mean()
data['rolling_std_3'] = data['value'].rolling(window=3).std()
data.dropna(inplace=True)

# Define features and target
X = data.drop('value', axis=1)
y = data['value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

def create_forecast_plot():
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='True Values'))
    fig.add_trace(go.Scatter(x=y_test.index, y=rf_pred, mode='lines', name='Random Forest Predictions'))
    fig.add_trace(go.Scatter(x=y_test.index, y=xgb_pred, mode='lines', name='XGBoost Predictions'))
    
    fig.update_layout(title='Time Series Forecasting',
                      xaxis_title='Date',
                      yaxis_title='Value')
    return fig

def create_feature_importance_plot(model, model_name):
    importance = model.feature_importances_
    features = X.columns
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=features, y=importance, name=model_name))
    
    fig.update_layout(title=f'{model_name} Feature Importances',
                      xaxis_title='Features',
                      yaxis_title='Importance')
    return fig

@app.get("/", response_class=HTMLResponse)
async def root():
    forecast_plot = create_forecast_plot().to_html(full_html=False)
    rf_importance_plot = create_feature_importance_plot(rf_model, "Random Forest").to_html(full_html=False)
    xgb_importance_plot = create_feature_importance_plot(xgb_model, "XGBoost").to_html(full_html=False)
    
    html_content = f"""
    <html>
        <head>
            <title>Time Series Forecasting Dashboard</title>
        </head>
        <body>
            <h1>Time Series Forecasting Dashboard</h1>
            <h2>Forecast Plot</h2>
            {forecast_plot}
            <h2>Random Forest Feature Importances</h2>
            {rf_importance_plot}
            <h2>XGBoost Feature Importances</h2>
            {xgb_importance_plot}
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
