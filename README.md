[![Build Status](https://app.travis-ci.com/LNshuti/oil-price-dashboard.svg?branch=main)](https://app.travis-ci.com/LNshuti/oil-price-dashboard)

# Oil Prices Dashboard
This repository attempts to reproduce Sean Taylor's tutorial on LineAPy. As an enchancement, I built a dashboard that tracks live global oil prices based on Sean's tutorial. The goal of building this project is to familiarize myself with unit testing, continuous integration/delivery, and building a full stack application with python. 


# Tools used

## Python 


```{python}
from statsforecast.models import AutoARIMA
from statsmodels.tsa.stattools import acf 
import pandas as pd 
import lineapy
import requests 
import re 
import numpy as np
from numpy.linalg import svd
import altair as alt
```

```{python}
%load_ext lineapy 
%load_ext nb_black
```

```{r}
uncertainty_plot = (
    forecast.pipe(alt.Chart, height=height, width=width)
    .encode(
        x="week",
        y=alt.Y(f"lo-{CI}", title="Price"),
        y2=alt.Y2(f"hi-{CI}", title="Price"),
    )
    .mark_area(opacity=0.2)
)

history_plot = (
    region_df.query(f"week >= '{plot_start_date}'")
    .pipe(alt.Chart, title=plot_title)
    .encode(x=alt.X("week", title="Week"), y=alt.Y("price", title="Price"))
    .mark_line()
)

forecast_plot = forecast.pipe(alt.Chart).encode(x="week", y="mean").mark_line()

cutoff_plot = (
    train.tail(1).pipe(alt.Chart).encode(x="week").mark_rule(strokeDash=[10, 2])
)

full_plot = uncertainty_plot + history_plot + forecast_plot + cutoff_plot
lineapy.save(full_plot, "gas_price_forecast")

```

```{python}
response = requests.get("https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls")
df = pd.read_excel(
    response.content,
    sheet_name="Data 12",
    index_col=0,
    skiprows=2,
    parse_dates=["Date"],
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
lineapy.save(df, "weekly_gas_price_data")
```

```{r}
raw_forecast = m_aa.predict(h=H, level=(CI,))
raw_forecast_exp = {key: np.exp(value) for key, value in raw_forecast.items()}
forecast = pd.DataFrame(raw_forecast_exp).assign(
    week=pd.date_range(train["week"].max(), periods=H, freq="W")
    + pd.Timedelta("7 days")
)
forecast = pd.concat(
    [
        forecast,
        train.tail(1)
        .rename(columns={"price": "mean"})
        .assign(**{f"lo-{CI}": lambda x: x["mean"], f"hi-{CI}": lambda x: x["mean"]}),
    ]
)
forecast.head()

```


```{r}
plots = []
for region in all_regions:
    result = forecast_region(
        region=region, cutoff_date=cutoff_date, height=200, width=200
    )
    plots.append(result["gas_price_forecast"])
```

```{r}
chart = alt.vconcat()
for i, plot in enumerate(plots):
    if i % 4 == 0:
        row = alt.hconcat()
        chart &= row
    row |= plot
chart
```
![visualization](https://user-images.githubusercontent.com/13305262/232337311-7086b21e-8929-4324-9818-c0bd792b8a62.png)



# References 
1. I used https://www.deploymachinelearning.com/ to learn about django and ML model deployment 
2. Sean Taylor. Lineapy Notebook. https://github.com/seanjtaylor/gas-price-forecast.git
