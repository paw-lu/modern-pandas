# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: modern-pandas
#     language: python
#     name: modern-pandas
# ---

# %% [markdown] Collapsed="false"
# # 4. Performance

# %% [markdown] Collapsed="false"
# ## Imports

# %% Collapsed="false"
# %matplotlib inline

import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

if int(os.environ.get("MODERN_PANDAS_EPUB", 0)):
    import prep  # noqa

sns.set_style("ticks")
sns.set_context("talk")
pd.options.display.max_rows = 10

# %% [markdown] Collapsed="true"
# ## Constructors

# %% [markdown] Collapsed="false"
# When combining many data sources—such as CDs—you can either:
#
# 1. Initialize one DataFrame and append to it
# 2. Make many small DataFrames and concatenate at the end
#
# Second method is faster.

# %% Collapsed="false"
# Python (bad) way
files = glob.glob("data/weather/*.csv")
columns = [
    "station",
    "date",
    "tmpf",
    "relh",
    "sped",
    "mslp",
    "p01i",
    "vsby",
    "gust_mph",
    "skyc1",
    "skyc2",
    "skyc3",
]
# init empty DataFrame, like you might for a list
weather = pd.DataFrame(columns=columns)
for fp in files:
    city = pd.read_csv(fp, usecols=columns)
    weather = weather.append(city)

# %% Collapsed="false"
# Pandas (good) way
files = glob.glob("data/weather/*.csv")
weather_dfs = [pd.read_csv(fp, names=columns, header=0) for fp in files]
weather = pd.concat(weather_dfs)

# %% [markdown] Collapsed="false"
# Can compare two different ways of building a DataFrame—`append_df` and `concat_df`.

# %% Collapsed="false"
import time

size_per = 5000
N = 100
cols = list("abcd")


def timed(n=30):
    """
    Running a microbenchmark. Never use this.
    """

    def deco(func):
        def wrapper(*args, **kwargs):
            timings = []
            for i in range(n):
                t0 = time.time()
                func(*args, **kwargs)
                t1 = time.time()
                timings.append(t1 - t0)
            return timings

        return wrapper

    return deco


@timed(60)
def append_df():
    """
    The pythonic (bad) way
    """
    df = pd.DataFrame(columns=cols)
    for _ in range(N):
        df.append(pd.DataFrame(np.random.randn(size_per, 4), columns=cols))
    return df


@timed(60)
def concat_df():
    """
    The pandorabe (good) way
    """
    dfs = [pd.DataFrame(np.random.randn(size_per, 4), columns=cols) for _ in range(N)]
    return pd.concat(dfs, ignore_index=True)


# %% Collapsed="false"
t_append = append_df()
t_concat = concat_df()

timings = (
    pd.DataFrame({"Append": t_append, "Concat": t_concat})
    .stack()
    .reset_index()
    .rename(columns={0: "Time (s)", "level_1": "Method"})
)
timings.head()

# %% Collapsed="false"
plt.figure(figsize=(4, 6))
sns.boxplot(x="Method", y="Time (s)", data=timings)
sns.despine()
plt.tight_layout()

# %% [markdown] Collapsed="true"
# ## Datatype

# %% [markdown] Collapsed="false"
# Pandas types are NumPys with some extensions—`categorical`, `datetime64`, and `timedelta64`. All columns should be same type. If that's not possible, it will have an `object` type.
#
# Traditional Pandas `NA` is a float. So if you have missing values with an integer column it will be an `object`. Text data will be an `object` column as well. There is no native `date` type—but it's usually okay to just use `datetime`.
#
# Good conversions methods are
#
# - `to_numeric`
# - `to_datetime`
# - `to_timedelta`

# %% [markdown] Collapsed="false"
# ## Iteration, apply, and vectorization

# %% [markdown] Collapsed="false"
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>.from_product</code></b> to get possible combinations of MultiIndex</li>
#     <li><b><code>.add_suffix</code></b> to add a suffix to column names</li>
#     <li><b><code>.reindex</code></b> to assign indexes to a DataFrame. Use level argument to assign MultiIndex to single-level indexes</li>
#     <li><b><code>.itertuples</code></b> for iterating over DataFrame rows as namedtuples.</li>
#     <li><b><code>.transform</code></b> is used for applying a function to a GroupBy object. Note that this will return values in the original DataFrame dimension.</li>
# </div>

# %% Collapsed="false"
df = pd.read_csv("data/307021124_T_ONTIME.csv")
delays = df["DEP_DELAY"]

# %% [markdown] Collapsed="false"
# Example: Get top values, _then_ sort.

# %% Collapsed="false"
delays.nlargest(5).sort_values()

# %% Collapsed="false"
delays.nsmallest(5).sort_values()

# %% Collapsed="false"
# %timeit delays.sort_values().tail(5)

# %% Collapsed="false"
# %timeit delays.nlargest(5).sort_values()

# %% Collapsed="false"
from utils import download_airports
import zipfile
if not os.path.exists("data/airports.csv.zip"):
    download_airports()

# %% Collapsed="false"
coord = (
    pd.read_csv(
        "data/airports.csv.zip",
        index_col=["AIRPORT"],
        usecols=["AIRPORT", "LATITUDE", "LONGITUDE"],
    )
    .groupby(level=0)
    .first()
    .dropna()
    .sample(n=500, random_state=42)
    .sort_index()
)

coord.head()

# %% Collapsed="false"
coord

# %% Collapsed="false"
coord.reindex(idx, level="origin")

# %% Collapsed="false"
idx = pd.MultiIndex.from_product([coord.index, coord.index], names=["origin", "dest"])
pairs = pd.concat(
    [
        coord.add_suffix("_1").reindex(idx, level="origin"),
        coord.add_suffix("_2").reindex(idx, level="dest"),
    ],
    axis=1
)
pairs.head()

# %% Collapsed="false"
idx = idx[idx.get_level_values(0) <= idx.get_level_values(1)]
len(idx)

# %% [markdown] Collapsed="false"
# Alternate implementation. Slower.

# %% Collapsed="false"
# Version not leveraging indexes
from itertools import product, chain

coord2 = coord.reset_index()
x = product(
    coord2.add_suffix("_1").itertuples(index=False),
    coord2.add_suffix("_2").itertuples(index=False),
)
y = [list(chain.from_iterable(z)) for z in x]

df2 = pd.DataFrame(
    y,
    columns=[
        "origin",
        "LATITUDE_1",
        "LONGITUDE_1",
        "dest",
        "LATITUDE_1",
        "LONGITUDE_2",
    ],
).set_index(["origin", "dest"])
df2.head()

# %% [markdown] Collapsed="false"
# "Python" vs NumPy

# %% Collapsed="false"
import math


def gcd_py(lat1, lng1, lat2, lng2):
    """
    Calculate great circle distance between two points.
    http://www.johndcook.com/blog/python_longitude_latitude/

    Parameters
    ----------
    lat1, lng1, lat2, lng2: float

    Returns
    -------
    distance:
      distance from ``(lat1, lng1)`` to ``(lat2, lng2)`` in kilometers.
    """
    # python2 users will have to use ascii identifiers (or upgrade)
    degrees_to_radians = math.pi / 180.0
    ϕ1 = (90 - lat1) * degrees_to_radians
    ϕ2 = (90 - lat2) * degrees_to_radians

    θ1 = lng1 * degrees_to_radians
    θ2 = lng2 * degrees_to_radians

    cos = math.sin(ϕ1) * math.sin(ϕ2) * math.cos(θ1 - θ2) + math.cos(ϕ1) * math.cos(ϕ2)
    # round to avoid precision issues on identical points causing ValueErrors
    cos = round(cos, 8)
    arc = math.acos(cos)
    return arc * 6373  # radius of earth, in kilometers


# %% Collapsed="false"
def gcd_vec(lat1, lng1, lat2, lng2):
    """
    Calculate great circle distance.
    http://www.johndcook.com/blog/python_longitude_latitude/

    Parameters
    ----------
    lat1, lng1, lat2, lng2: float or array of float

    Returns
    -------
    distance:
      distance from ``(lat1, lng1)`` to ``(lat2, lng2)`` in kilometers.
    """
    # python2 users will have to use ascii identifiers
    ϕ1 = np.deg2rad(90 - lat1)
    ϕ2 = np.deg2rad(90 - lat2)

    θ1 = np.deg2rad(lng1)
    θ2 = np.deg2rad(lng2)

    cos = np.sin(ϕ1) * np.sin(ϕ2) * np.cos(θ1 - θ2) + np.cos(ϕ1) * np.cos(ϕ2)
    arc = np.arccos(cos)
    return arc * 6373


# %% [markdown] Collapsed="false"
# Iteration vs `.apply`

# %% Collapsed="false"
pairs.itertuples

# %% Collapsed="false"
# %%time
# Iteration
pd.Series([gcd_py(*x) for x in pairs.itertuples(index=False)], index=pairs.index)

# %% Collapsed="false"
# %%time
# ``.apply``
r = pairs.apply(
    lambda x: gcd_py(
        x["LATITUDE_1"], x["LONGITUDE_1"], x["LATITUDE_2"], x["LONGITUDE_2"]
    ),
    axis=1,
);

# %% [markdown] Collapsed="false"
# `.apply` is _way_ slower. You rarely want to use `.apply` and even less so with `axis=1`. Write functions that take in arrays and pass those directly.

# %% Collapsed="false"
# %%time
# Best version
r = gcd_vec(
    pairs["LATITUDE_1"], pairs["LONGITUDE_1"], pairs["LATITUDE_2"], pairs["LONGITUDE_2"]
)

# %% [markdown] Collapsed="false"
# Some operations are a little bit tricker to vectorize.

# %% Collapsed="false"
import random


def create_frame(n, n_groups):
    # just setup code, not benchmarking this
    stamps = pd.date_range("20010101", periods=n, freq="ms")
    random.shuffle(stamps.values)
    return pd.DataFrame(
        {
            "name": np.random.randint(0, n_groups, size=n),
            "stamp": stamps,
            "value": np.random.randint(0, n, size=n),
            "value2": np.random.randn(n),
        }
    )


df = create_frame(1000000, 10000)


def f_apply(df):
    # Typical transform
    return df.groupby("name").value2.apply(lambda x: (x - x.mean()) / x.std())


def f_unwrap(df):
    # "unwrapped"
    g = df.groupby("name").value2
    v = df.value2
    return (v - g.transform(np.mean)) / g.transform(np.std)


# %% Collapsed="false"
# %timeit f_apply(df)

# %% Collapsed="false"
# %timeit f_unwrap(df)

# %% [markdown] Collapsed="false"
# Pandas `GroupBy` will intercept calls for common functions like mean, std, and sum and use optimized Cython versions. Note that `GroupBy.apply` exists for flexibility, but is not the fastest.

# %% [markdown] Collapsed="false"
# ## Categoricals

# %% [markdown] Collapsed="false"
# Good way to represent strings with few unique values.

# %% Collapsed="false"
import string

s = pd.Series(np.random.choice(list(string.ascii_letters), 100000))
print("{:0.2f} KB".format(s.memory_usage(index=False) / 1000))

# %% Collapsed="false"
c = s.astype("category")
print("{:0.2f} KB".format(c.memory_usage(index=False) / 1000))

# %% Collapsed="false"
