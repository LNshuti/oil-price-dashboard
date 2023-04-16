[![Build Status](https://app.travis-ci.com/LNshuti/oil-price-dashboard.svg?branch=main)](https://app.travis-ci.com/LNshuti/oil-price-dashboard)


# Oil Prices Dashboard
This repository uses software engineering best practices to build a dashboard that tracks live global oil prices. The goal of building this project is to familiarize myself with unit testing, continuous integration/delivery, and building a full stack application with python. 


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

# References 
1. I used https://www.deploymachinelearning.com/ to learn about django and ML model deployment 
