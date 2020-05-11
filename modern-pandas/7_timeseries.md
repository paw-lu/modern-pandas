# -*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: modern-pandas
    language: python
    name: modern-pandas
---

# 7 Timeseries

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
## Imports
<!-- #endregion -->

```python
%matplotlib inline

import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='ticks', context='talk')
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
## Data
<!-- #endregion -->

```python
gs = web.DataReader("GS", data_source="yahoo", start="2006-01-01", end="2010-01-01")
gs.head().round(2)
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
## Special slicing
<!-- #endregion -->

<div class="alert alert-block alert-info">
    <b>Techniques of note</b>
    <br><br>
    <li><b><code>df.loc["2006"]</code></b> can slice using strings for dates</li>
</div>


Timestamp slices have a special feature where you can slice using strings

```python
gs.index[0]
```

```python
# Bad
# Works—but a pain to write
gs.loc[pd.Timestamp('2006-01-01'):pd.Timestamp('2006-12-31')].head()
```

```python
# Good!
gs.loc["2006"].head()
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
## Special methods
<!-- #endregion -->

<div class="alert alert-block alert-info">
    <b>Techniques of note</b>
    <br><br>
    <li><b><code>.resample</code></b> Can split time series into groups by <i>time frames</i> and apply a function to each group. Returns a grouped shape.</li>
    <li><b><code>.rolling</code></b> Can split time series into groups by <i>rows</i> and apply a function to each group. Returns the same shape.</li>
    <li><b><code>.expanding</code></b> Can apply a function to all rows before it.</li>
    <li><b><code>.ewm</code></b> To apply exponentially weighted functions.</li>
</div>

```python
gs.resample("5d").mean().head()
```

```python
gs.resample("W").agg(["mean", "sum"]).head()
```

```python
gs.resample("6H").mean().head()
```

```python
gs.Close.plot(label='Raw')
gs.Close.rolling(28).mean().plot(label='28D MA')
gs.Close.expanding().mean().plot(label='Expanding Average')
gs.Close.ewm(alpha=0.03).mean().plot(label='EWMA($\\alpha=.03$)')

plt.legend(bbox_to_anchor=(1.25, .5))
plt.ylabel("Close ($)")
sns.despine();
```

```python
roll = gs.Close.rolling(30, center=True)
roll
```

```python
m = roll.agg(["mean", "std"])
ax = m["mean"].plot()
ax.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.25)
plt.tight_layout()
plt.ylabel("Close ($)")
sns.despine()
```

<!-- #region toc-hr-collapsed=true toc-nb-collapsed=true -->
## Grab bag
<!-- #endregion -->

<div class="alert alert-block alert-info">
    <b>Techniques of note</b>
    <br><br>
    <li><b><code>DateOffset</code></b> Add an offset to dates.</li>
    <li><b><code>tseries.holiday</code></b> To get holidays.</li>
    <li><b><code>tz_localize</code></b> To assign a timezone.</li>
    <li><b><code>tz_convert</code></b> To convert to different timezone.</li>
</div>


### Offsets

```python
gs.index
```

```python
gs.index + pd.DateOffset(months=3, days=-2)
```

### Holiday calendar

```python
from pandas.tseries.holiday import USColumbusDay
```

```python
USColumbusDay.dates("2015-01-01", "2020-01-01")
```

### Timezones

```python
gs.tz_localize('US/Eastern').tz_convert('UTC').head()
```

## Modeling time series


Predict average monthly flights


<div class="alert alert-block alert-info">
    <b>Techniques of note</b>
    <br><br>
    <li><b><code>.shift</code></b> To add lagged values.</li>
</div>


### Data

```python
import os
import io
import glob
import zipfile
from utils import download_timeseries

import statsmodels.api as sm


def download_many(start, end):
    months = pd.period_range(start, end=end, freq="M")
    # We could easily parallelize this loop.
    for i, month in enumerate(months):
        download_timeseries(month)


def time_to_datetime(df, columns):
    """
    Combine all time items into datetimes.

    2014-01-01,1149.0 -> 2014-01-01T11:49:00
    """

    def converter(col):
        timepart = (
            col.astype(str)
            .str.replace("\.0$", "")  # NaNs force float dtype
            .str.pad(4, fillchar="0")
        )
        return pd.to_datetime(
            df["fl_date"]
            + " "
            + timepart.str.slice(0, 2)
            + ":"
            + timepart.str.slice(2, 4),
            errors="coerce",
        )
        return datetime_part

    df[columns] = df[columns].apply(converter)
    return df


def read_one(fp):
    df = (
        pd.read_csv(fp, encoding="latin1")
        .rename(columns=str.lower)
        .drop("unnamed: 6", axis=1)
        .pipe(
            time_to_datetime, ["dep_time", "arr_time", "crs_arr_time", "crs_dep_time"]
        )
        .assign(fl_date=lambda x: pd.to_datetime(x["fl_date"]))
    )
    return df
```

```python
def unzip_one(zip_file):
    """Written by me."""
    dir_name = zip_file.replace(".zip", "")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dir_name)
    file = glob.glob(f"{dir_name}/*.csv")[0]
    return file
```

```python
store = "data/ts.hdf5"

if not os.path.exists(store):
    download_many("2000-01-01", "2016-01-01")

    zips = glob.glob(os.path.join("data", "timeseries", "*.zip"))
    csvs = [unzip_one(fp) for fp in zips]
    dfs = [read_one(fp) for fp in csvs]
    df = pd.concat(dfs, ignore_index=True)

    df["origin"] = df["origin"].astype("category")
    df.to_hdf(store, "ts", format="table")
else:
    df = pd.read_hdf(store, "ts")
```

```python
df.dtypes
```

### Modeling


First find historic values (average monthly flights)

```python
daily = df.fl_date.value_counts().sort_index()
y = daily.resample("MS").mean()
y.head()
```

```python
ax = y.plot()
ax.set(ylabel="Average monthly flights")
sns.despine()
```

```python
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
```

With time series,
most important features are the previous
—or _lagged_—
values.

Start by manually running regression on lagged values of itself.
This will suffer from:

- Multicollinearity
- Autocorrelation
- Non-stationary
- Seasonality

```python
X = pd.concat(
    [y.shift(i) for i in range(6)],
    axis=1,
    keys=["y"] + ["L%s" % i for i in range(1, 6)],
).dropna()
X.head()
```

Now fit to the lagged values using statsmodels.

```python
mod_lagged = smf.ols(
    "y ~ trend + L1 + L2 + L3 + L4 + L5", data=X.assign(trend=np.arange(len(X)))
)
res_lagged = mod_lagged.fit()
res_lagged.summary()
```

#### Problems


##### Multicollinearity.


Correlation between different columns.

```python
sns.heatmap(X.corr())
```

You would expect
coefficients to gradually decline to zero.
Most recent data is most important.
Second most recent data is second most important.
etc

```python
ax = res_lagged.params.drop(['Intercept', 'trend']).plot.bar(rot=0)
plt.ylabel('Coefficeint')
sns.despine()
```

##### Autocorrelation


Pattern in the residuals of your regression.
Residuals should be white noise.
In this case
if a residual at time `t` was above expectation,
than residual at `t + 1` is much more likely
to be above average as well.

```python
# `Results.resid` is a Series of residuals: y - ŷ
mod_trend = sm.OLS.from_formula(
    "y ~ trend", data=y.to_frame(name="y").assign(trend=np.arange(len(y)))
)
res_trend = mod_trend.fit()
```

```python
def tsplot(y, lags=None, figsize=(10, 8)):
    """Plot residuals."""
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax
tsplot(res_trend.resid, lags=36)
```

The top subplot shows the time series of residual $e_t$—
which should be white noise.

The bottom shows autocorrelation
of the residuals as a correlogram.
It measures the correlation between a value
and its lagged self.

The bottom right plot is partial autocorrelation.


##### Stationary


Stationary data
means that the
variance, mean, and autocorrelation structure
do not change over time.

Typically this means a flat-looking series,
without trend.

One way to handle non-stationary data
is to difference the non-stationary variable
until it is stationary.

```python
y.to_frame(name="y").assign(Δy=lambda x: x.y.diff()).plot(subplots=True)
sns.despine()
```

One way to quantify
whether a series is non-stationary
is the **Augmented Dickey-Fuller** test.

The null hypothesis in the test
is that the data is non-stationary,
and therefore needs to be differenced.

The alternate hypothesis is that is is stationary,
and therefore does not need to be differenced.

This test is available in `smt.adfuller` in stastmodels.

```python
from collections import namedtuple

ADF = namedtuple("ADF", "adf pvalue usedlag nobs critical icbest")
ADF(*smt.adfuller(y))._asdict()
```

So here we failed to reject the null hypothesis.
Difference it and try again.

```python
ADF(*smt.adfuller(y.diff().dropna()))._asdict()
```

Now fit another OLS model.

```python
data = (
    y.to_frame(name="y")
    .assign(Δy=lambda df: df.y.diff())
    .assign(LΔy=lambda df: df.Δy.shift())
)
mod_stationary = smf.ols("Δy ~ LΔy", data=data.dropna())
res_stationary = mod_stationary.fit()
tsplot(res_stationary.resid, lags=24);
```

##### Seasonality


We have a strong monthly seasonality.

```python
smt.seasonal_decompose(y).plot()
```

#### ARIMA


ARIMA can handle all the problems specified:

- Multicollinearity
- Autocorrelation
- Non-stationary
- Seasonality

**A**utoRegressive  
**I**ntegrated  
**M**oving  
**A**verage  


##### **A**utoRegressive


Predict a variable
by a linear combination
of its lagged values

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \ldots + \phi_p y_{t-p} + e_t$$

Where `p` is the number of lagged values and `c` is a constant and `e` is white noise.


##### **I**ntegrated


The opposite of differencing.
Deals with stationarity.
If you have to difference your dataset 1 time to get it stationary, then $d=1$.

$\Delta y_t = y_t - y_{t-1}$ for $d=1$.


##### **M**oving **A**verage


$$y_t = c + e_t + \theta_1 e_{t-1} + \theta_2 e_{t-2} + \ldots + \theta_q e_{t-q}$$

Here the coefficients are residuals from previous predictions. 


##### Combine


$$\Delta y_t = c + \phi_1 \Delta y_{t-1} + \theta_t e_{t-1} + e_t$$

Using lag notation, where $L y_t = y_{t-1}$, i.e. y.shift() in pandas, we can rewrite that as

$$(1 - \phi_1 L) (1 - L)y_t = c + (1 + \theta L)e_t$$

for our specific `ARIMA(1, 1, 1)` model

```python
mod = smt.SARIMAX(y, trend='c', order=(1, 1, 1))
res = mod.fit()
tsplot(res.resid[2:], lags=24);
```

```python
res.summary()
```

Looks better,
but still needs seasonality adjustment.

Seasonal ARIMA model is written as
$\mathrm{ARIMA}(p,d,q)×(P,D,Q)_s$.
Lowercase letters are non-seasonal components.
Upper-case letters are similar specification for seasonal component—
where $s$ is the periodicity
(4 quarterly, 12 monthly).

We multiply the two processes—
one for seasonal
and one for non-seasonal—
together.

So the nonseasonal component is

- $p=1$: period autoregressive: use $y_{t-1}$
- $d=1$: one first-differencing of the data (one month)
- $q=2$: use the previous two non-seasonal residual, $e_{t-1}$     $e_{t-2}$, to forecast

And the seasonal component is

- $P=0$: Don't use any previous seasonal values
- $D=1$: Difference the series 12 periods back: y.diff(12)
- $Q=2$: Use the two previous seasonal residuals

```python
mod_seasonal = smt.SARIMAX(
    y,
    trend="c",
    order=(1, 1, 2),
    seasonal_order=(0, 1, 2, 12),
    simple_differencing=False,
)
res_seasonal = mod_seasonal.fit()
res_seasonal.summary()
```

```python
tsplot(res_seasonal.resid[12:], lags=24);
```

Looks much better!

In reality user grid search to find parameters.
Optimize for AIC or BIC.

Tips [on this blog post](https://otexts.com/fpp2/arima-r.html) and [this stackoverflow answer](https://stackoverflow.com/questions/22770352/auto-arima-equivalent-for-python/22770973#22770973).


### Forecasting


Do one step ahead forecast.
At each point (month),
we take the history up to that point
and make a forecast.

```python
pred = res_seasonal.get_prediction(start="2001-03-01")
pred_ci = pred.conf_int()
ax = y.plot(label="observed")
pred.predicted_mean.plot(ax=ax, label="Forecast", alpha=0.7)
ax.fill_between(
    pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="k", alpha=0.2
)
ax.set_ylabel("Monthly Flights")
plt.legend()
sns.despine()
```

There are a few places
where the observed data goes outside
the 95% confidence interval.

We can make dynamic forecasts
as of some month.
The forecast from that point forward
will only use information available as of January 2013.

It still work similar to the one-step process above.
But instead of predicting on actual values
after January 2013,
you predict on the forecast values.

```python
pred_dy = res_seasonal.get_prediction(start="2002-03-01", dynamic="2013-01-01")
pred_dy_ci = pred_dy.conf_int()
ax = y.plot(label="observed")
pred_dy.predicted_mean.plot(ax=ax, label="Forecast")
ax.fill_between(
    pred_dy_ci.index,
    pred_dy_ci.iloc[:, 0],
    pred_dy_ci.iloc[:, 1],
    color="k",
    alpha=0.25,
)
ax.set_ylabel("Monthly Flights")

# Highlight the forecast area
ax.fill_betweenx(
    ax.get_ylim(), pd.Timestamp("2013-01-01"), y.index[-1], alpha=0.1, zorder=-1
)
ax.annotate("Dynamic $\\longrightarrow$", (pd.Timestamp("2013-02-01"), 550))

plt.legend()
sns.despine()
```
```python

```

