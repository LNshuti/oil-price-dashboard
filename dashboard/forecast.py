"""
Monthly price forecasting.

Every outlook runs in two passes:

1. Backtest: fit on all but the last ``horizon`` months and score each
   model on the held-out months. This picks the recommended model.
2. Forecast: refit on the full history and predict ``horizon`` months
   past the last complete month, with 80% and 95% intervals.
"""

import threading
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from xgboost import XGBRegressor

MODELS = ("AutoARIMA", "AutoETS", "LightGBM", "XGBoost")
LEVELS = (80, 95)
HORIZONS = (3, 6, 12, 18, 24)
TRAIN_YEARS = 20
CONFORMAL_WINDOWS = 3


@dataclass(frozen=True)
class Outlook:
    history: pd.Series      # complete monthly means used for training
    backtest: pd.DataFrame  # ds, actual, one column per model
    metrics: pd.DataFrame   # Model, MAE, RMSE, MAPE (sorted, best first)
    forecast: pd.DataFrame  # ds, model columns and -lo-/-hi- intervals
    recommended: str
    horizon: int

    def band(self, model: str, level: int) -> tuple[pd.Series, pd.Series]:
        return (self.forecast[f"{model}-lo-{level}"],
                self.forecast[f"{model}-hi-{level}"])


def _to_nixtla(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "unique_id": "price",
        "ds": series.index,
        "y": series.to_numpy(dtype=float),
    })


def _fit_predict(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    sf = StatsForecast(
        models=[AutoARIMA(season_length=12), AutoETS(season_length=12)],
        freq="MS",
        n_jobs=1,
    )
    sf_preds = sf.forecast(df=train, h=horizon, level=list(LEVELS))

    mlf = MLForecast(
        models={
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.1, num_leaves=31,
                verbose=-1,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                verbosity=0,
            ),
        },
        freq="MS",
        lags=[1, 2, 3, 6, 12],
        lag_transforms={
            1: [ExpandingMean()],
            3: [RollingMean(window_size=3)],
        },
    )
    # Conformal intervals give the ML models the same 80/95% bands.
    mlf.fit(
        train,
        prediction_intervals=PredictionIntervals(
            n_windows=CONFORMAL_WINDOWS, h=horizon,
        ),
    )
    ml_preds = mlf.predict(h=horizon, level=list(LEVELS))

    merged = sf_preds.merge(ml_preds, on=["unique_id", "ds"])
    return merged.drop(columns="unique_id")


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(np.mean(np.abs(error / actual)) * 100),
    }


def build_outlook(monthly_prices: pd.Series, horizon: int) -> Outlook:
    """Backtest every model, then forecast forward from all history."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    start = monthly_prices.index.max() - pd.DateOffset(years=TRAIN_YEARS)
    history = monthly_prices[monthly_prices.index > start]
    full = _to_nixtla(history)
    train, test = full.iloc[:-horizon], full.iloc[-horizon:]

    held_out = _fit_predict(train, horizon)
    backtest = pd.DataFrame({"ds": test["ds"].to_numpy(),
                             "actual": test["y"].to_numpy()})
    rows = []
    for model in MODELS:
        backtest[model] = held_out[model].to_numpy()
        rows.append({"Model": model,
                     **score(backtest["actual"].to_numpy(),
                             backtest[model].to_numpy())})
    metrics = pd.DataFrame(rows).sort_values(["MAPE", "MAE"])
    metrics = metrics.reset_index(drop=True)

    forecast = _fit_predict(full, horizon)
    return Outlook(
        history=history,
        backtest=backtest,
        metrics=metrics,
        forecast=forecast,
        recommended=metrics.loc[0, "Model"],
        horizon=horizon,
    )


_cache: dict = {}
_cache_lock = threading.Lock()


def cached_outlook(key: tuple, monthly_prices: pd.Series,
                   horizon: int) -> Outlook:
    """Memoize outlooks by (fuel, area, horizon, data as-of date)."""
    with _cache_lock:
        if key not in _cache:
            _cache[key] = build_outlook(monthly_prices, horizon)
        return _cache[key]
