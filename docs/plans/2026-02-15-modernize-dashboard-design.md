# Oil Price Dashboard Modernization Design

## Overview

Modernize the oil price forecasting dashboard with state-of-the-art tabular ML models, Tufte-themed visualizations, NEJM-formatted tables, and deploy as an enterprise-grade Gradio app on HuggingFace Spaces.

## Approach

Single-file Gradio app (`app.py`) deployed directly to HuggingFace Spaces. Simple structure optimized for HF deployment while maintaining clean code organization via classes/functions.

## Data Pipeline

- **Source:** EIA petroleum data (same as current)
- **Refresh:** On-demand with 24-hour cache
- **Frequencies:** Weekly and monthly support

## Forecasting Models (Nixtla Stack)

| Model | Library | Purpose |
|-------|---------|---------|
| AutoARIMA | StatsForecast | Statistical baseline |
| AutoETS | StatsForecast | Exponential smoothing |
| LightGBM | MLForecast | Gradient boosting with lag features |
| XGBoost | MLForecast | Alternative gradient boosting |

**Output:** Point forecasts + 80%/95% confidence intervals, configurable horizon (4, 8, 12, 26, 52 weeks).

## Visualization

### Tufte Theme (Charts)
- No gridlines, no box frames
- Maximized data-ink ratio
- Muted palette (grays + one accent)
- Direct labeling over legends
- Small multiples for comparison

### NEJM Format (Tables)
- Horizontal rules only (top, header-bottom, table-bottom)
- No vertical lines
- Right-aligned numbers, left-aligned text
- Subtle alternating row shading

### Charts
1. Historical prices + forecast with confidence bands
2. Model comparison overlay
3. Feature importance (ML models)
4. Residual diagnostics (expandable)

## Gradio App Structure

```
┌─────────────────────────────────────────────────────┐
│  Oil Price Forecasting Dashboard                    │
├─────────────────────────────────────────────────────┤
│ [Refresh Data] [Forecast Horizon ▼] [Run Forecast] │
├─────────────────────────────────────────────────────┤
│  Tab: Forecast │ Tab: Model Comparison │ Tab: ...  │
├────────────────┴────────────────────────────────────┤
│  Tufte-styled chart                                 │
│  NEJM-styled table                                  │
│  [Download CSV] [Download PNG]                      │
└─────────────────────────────────────────────────────┘
```

**Tabs:**
1. Forecast - Main view with chart + table
2. Model Comparison - Accuracy metrics (MAE, RMSE, MAPE)
3. Data Explorer - Raw data with filtering
4. About - Methodology and sources

## File Structure

```
oil-price-dashboard/
├── app.py                    # New: Gradio app
├── requirements.txt          # Modified: gradio, nixtla libs
├── tests/
│   └── test_app.py          # New: app tests
└── README.md                # Modified: HF Spaces badge
```

## Testing

- Unit tests for data fetching (mocked)
- Unit tests for forecast pipeline (sample data)
- Smoke test: app launches without error
- Existing tests pass (no regressions)

## Git Workflow

1. Create branch `feat-modernize` from `main`
2. Implement in logical commits
3. Run pytest to verify
4. Test locally with `python app.py`
5. Create PR to `main`

## Deployment

HuggingFace Spaces with:
- `app.py` + `requirements.txt`
- Custom CSS embedded for styling
- README.md dataset card
