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

# References 
1. I used https://www.deploymachinelearning.com/ to learn about django and ML model deployment 
