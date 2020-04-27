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
# # 2. Method chaining

# %% [markdown] Collapsed="false"
# Many chaining-friendly methods such as:
#
# - `.assign`
# - `.pipe`
# - `.rename`
# - Window methods: `.rolling`
# - `.resample`
# - `.where` and `mask` and Indexers accept lambdas

# %% [markdown] Collapsed="false"
# ## Data

# %% Collapsed="false"
# %matplotlib inline

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import prep
import seaborn as sns

# sns.set(style='ticks', context='talk')

# %% Collapsed="false"
def read(fp):
    df = (
        pd.read_csv(fp)
        .rename(columns=str.lower)
#         .drop("unnamed: 36", axis=1)
        .pipe(extract_city_name)
        .pipe(
            time_to_datetime, ["dep_time", "arr_time", "crs_arr_time", "crs_dep_time"]
        )
        .assign(
            fl_date=lambda x: pd.to_datetime(x["fl_date"]),
            dest=lambda x: pd.Categorical(x["dest"]),
            origin=lambda x: pd.Categorical(x["origin"]),
            tail_num=lambda x: pd.Categorical(x["tail_num"]),
            unique_carrier=lambda x: pd.Categorical(x["unique_carrier"]),
            cancellation_code=lambda x: pd.Categorical(x["cancellation_code"]),
        )
    )
    return df


def extract_city_name(df):
    """
    Chicago, IL -> Chicago for origin_city_name and dest_city_name
    """
    cols = ["origin_city_name", "dest_city_name"]
    city = df[cols].apply(lambda x: x.str.extract("(.*), \w{2}", expand=False))
    df = df.copy()
    df[["origin_city_name", "dest_city_name"]] = city
    return df


def time_to_datetime(df, columns):
    """
    Combine all time items into datetimes.

    2014-01-01,0914 -> 2014-01-01 09:14:00
    """
    df = df.copy()

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

    df[columns] = df[columns].apply(converter)
    return df


output = "data/flights.h5"

if not os.path.exists(output):
    df = read("data/307021124_T_ONTIME.csv")
    df.to_hdf(output, "flights", format="table")
else:
#     df = read("data/307021124_T_ONTIME.csv")
    df = pd.read_hdf(output, "flights")
df.info()

# %% [markdown] Collapsed="false"
# ## Inspecting chains

# %% [markdown] Collapsed="false"
# [Tom Augspurger](https://tomaugspurger.github.io/method-chaining) wrote some decorators you can use to inspect and log DataFrame properties in between chain steps.

# %% Collapsed="false"
from functools import wraps
import logging

def log_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.info("%s,%s" % (func.__name__, result.shape))
        return result
    return wrapper

def log_dtypes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.info("%s,%s" % (func.__name__, result.dtypes))
        return result
    return wrapper


@log_shape
@log_dtypes
def load(fp):
    df = pd.read_csv(fp, index_col=0, parse_dates=True)

@log_shape
@log_dtypes
def update_events(df, new_events):
    df.loc[new_events.index, 'foo'] = new_events
    return df


# %% [markdown] Collapsed="false"
# ## Inplace argument 

# %% [markdown] Collapsed="false"
# Many Pandas methods have an `inplace` keyword that is `False` by default. Keep it that way.

# %% [markdown] Collapsed="false"
# ## Examples

# %% Collapsed="false"
df.head()

# %% Collapsed="false"
(
    # Drop rows with missing values in "dep_time" or "unique_carrier"
    df.dropna(subset=["dep_time", "unique_carrier"])
    # Look only at the top 5 most popular carriers
    .loc[
        lambda _df: _df["unique_carrier"].isin(
            _df["unique_carrier"].value_counts().index[:5]
        )
    ]
    # Set the time as the index
    .set_index("dep_time")
    # Group by both the unique carrier and by each our on the index
    .groupby(["unique_carrier", pd.Grouper(freq="H")])
    # Get the number of "fl_num", this is the total number of flight
    .fl_num.count()
    # Pivot the table so the index values become the columns
    .unstack("unique_carrier")
    # My addition. Since column is categorical we need to drop all these
    # empty columns
    .loc[:, lambda _df: _df.columns[0 < _df.sum()]]
    # Fill missing values with 0
    .fillna(0)
    # Get the sum from the previous 24 rows
    .rolling(24).sum()
    # Rename
    .rename_axis("Flights per Day", axis="columns")
    # Plot
    .plot()
)

# %% Collapsed="false"
fig, ax = plt.subplots(figsize=(15, 5))
ax.set_ylim(-50, 50)
sns.despine()
(
    df[["fl_date", "tail_num", "dep_time", "dep_delay"]]
    .dropna()
    .sort_values("dep_time")
    .loc[lambda _df: _df.dep_delay < 500]
    .assign(
        turn=lambda _df: _df.groupby(["fl_date", "tail_num"])
        .dep_time.transform("rank")
        .astype(int)
    )
    # Pipe is powerful! Use a string as a keyword here if the dataframe
    # is not the first argument of the function
    # Note we can also pass **kwargs
    .pipe((sns.boxplot, "data"), x="turn", y="dep_delay", ax=ax)
)

# %% Collapsed="false"
plt.figure(figsize=(15, 5))
(
    df[["fl_date", "tail_num", "dep_time", "dep_delay"]]
    .dropna()
    .assign(hour=lambda x: x.dep_time.dt.hour)
    .query("5 < dep_delay < 600")
    .pipe((sns.boxplot, "data"), "hour", "dep_delay")
)
sns.despine()

# %% [markdown] Collapsed="false"
# I liked this example from [Chang Hsin Lee](https://changhsinlee.com/pyjanitor/) on using the `lambda` function to help with method chaining.
#
# ```python
# import pandas as pd
#
# raw_data = pd.read_csv("sales.csv")
# df = (
#     raw_data.assign(order_date=lambda _df: pd.to_datetime(_df["order_date"]))[
#         lambda _df: _df["order_date"] >= pd.to_datetime("2019-11-09")
#     ]
#     .groupby(["category"], axis=1)
#     .agg({"dollars": "sum", "volume": "sum",})
# )
# ```

# %% [markdown] Collapsed="false"
# Another good method chaining example from [dataschool](https://www.dataschool.io/future-of-pandas/#methodchaining) with using `lambda` with `.assign`
#
# ```python
# import pandas as pd
#
# (
#     pd.read_csv("data/titanic.csv.gz")
#     .query("Age < Age.quantile(.99)")
#     .assign(
#         Sex=lambda df: df["Sex"].replace({"female": 1, "male": 0}),
#         Age=lambda df: pandas.cut(
#             df["Age"].fillna(df.Age.median()),
#             bins=[df.Age.min(), 18, 40, df.Age.max()],
#             labels=["Underage", "Young", "Experienced"],
#         ),
#     )
#     .pivot_table(values="Sex", columns="Pclass", index="Age", aggfunc="mean")
#     .rename_axis("", axis="columns")
#     .rename("Class {}".format, axis="columns")
#     .style.format("{:.2%}")
# )
# ```

# %% Collapsed="false"
