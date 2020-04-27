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
# # 3. Indexes

# %% [markdown] Collapsed="false"
# ## Data

# %% Collapsed="false"
# %matplotlib inline

import datetime
import glob
import json
from io import StringIO

import numpy as np
import pandas as pd
import requests

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

# States are broken into networks. The networks have a list of ids, each representing a station.
# We will take that list of ids and pass them as query parameters to the URL we built up ealier.
states = """AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME
 MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
 WA WI WV WY""".split()

# IEM has Iowa AWOS sites in its own labeled network
networks = ["AWOS"] + ["{}_ASOS".format(state) for state in states]


# %% Collapsed="false"
def get_weather(
    stations, start=pd.Timestamp("2014-01-01"), end=pd.Timestamp("2014-01-31")
):
    """
    Fetch weather data from MESONet between ``start`` and ``stop``.
    """
    url = (
        "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        "&data=tmpf&data=relh&data=sped&data=mslp&data=p01i&data=vsby&data=gust_mph&data=skyc1&data=skyc2&data=skyc3"
        "&tz=Etc/UTC&format=comma&latlon=no"
        "&{start:year1=%Y&month1=%m&day1=%d}"
        "&{end:year2=%Y&month2=%m&day2=%d}&{stations}"
    )
    stations = "&".join("station=%s" % s for s in stations)
    weather = (
        pd.read_csv(url.format(start=start, end=end, stations=stations), comment="#")
        .rename(columns={"valid": "date"})
        .rename(columns=str.strip)
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .set_index(["station", "date"])
        .sort_index()
    )
    float_cols = ["tmpf", "relh", "sped", "mslp", "p01i", "vsby", "gust_mph"]
    weather[float_cols] = weather[float_cols].apply(pd.to_numeric, errors="coerce")
    return weather


def get_ids(network):
    url = "http://mesonet.agron.iastate.edu/geojson/network.php?network={}"
    r = requests.get(url.format(network))
    md = pd.json_normalize(r.json()["features"])
    md["network"] = network
    return md


# %% [markdown] Collapsed="false"
# ## `json_normalize`

# %% Collapsed="false"
url = "http://mesonet.agron.iastate.edu/geojson/network.php?network={}"
r = requests.get(url.format("AWOS"))
js = r.json()
js["features"][:2]

# %% [markdown] Collapsed="false"
# This won't work

# %% Collapsed="false"
pd.DataFrame(js["features"]).head()

# %% [markdown] Collapsed="false"
# Use `json_normalize`

# %% Collapsed="false"
pd.json_normalize(js['features']).head()

# %% Collapsed="false"
import os

ids = pd.concat([get_ids(network) for network in networks], ignore_index=True)
gr = ids.groupby("network")

store = "data/weather.h5"

if not os.path.exists(store):
    os.makedirs("data/weather", exist_ok=True)

    for k, v in gr:
        weather = get_weather(v["id"])
        weather.to_csv("data/weather/{}.csv".format(k))

    weather = pd.concat(
        [
            pd.read_csv(f, parse_dates=["date"], index_col=["station", "date"])
            for f in glob.glob("data/weather/*.csv")
        ]
    ).sort_index()

    weather.to_hdf("data/weather.h5", "weather")
else:
    weather = pd.read_hdf("data/weather.h5", "weather")

# %% Collapsed="false"
weather.head()

# %% Collapsed="false"
airports = ["W43", "AFO", "82V", "DUB"]
g = sns.FacetGrid(
    weather.loc[airports].reset_index(),
    col="station",
    hue="station",
    col_wrap=2,
    height=4,
)
g.map(sns.regplot, "sped", "gust_mph")

# %% [markdown] Collapsed="false"
# ## Set operations

# %% [markdown] Collapsed="false"
# Find airports that have both weather and flight information.

# %% [markdown] Collapsed="false"
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>MultiIndex.level</code><b> for grabbing specific indexes from a MultiIndex</li>
#     <li><b><code>Index.difference</code><b> for finding indexes that exist in one set but not the other</li>
#     <li><b><code>^</code><b> for symmetric differences</li>
# </div>

# %% Collapsed="false"
flights = pd.read_hdf("data/flights.h5", "flights")
weather_locs = weather.index.levels[0]
# The `categories` attributes of a Categorical is an Index
origin_locs = flights.origin.cat.categories
dest_locs = flights.dest.cat.categories

airports = weather_locs & origin_locs & dest_locs
airports

# %% Collapsed="false"
print("Weather, no flights:\n\t", weather_locs.difference(origin_locs | dest_locs), end='\n\n')

print("Flights, no weather:\n\t", (origin_locs | dest_locs).difference(weather_locs), end='\n\n')

print("Dropped Stations:\n\t", (origin_locs | dest_locs) ^ weather_locs)

# %% [markdown] Collapsed="false"
# ## Flavors

# %% [markdown] Collapsed="false"
# Pandas has multiple type of indexes:
#
# 1. `Index`
# 2. `Int64Index`
# 3. `RangeIndex` (Memory-saving special case of Int64Index)
# 4. `FloatIndex`
# 5. `DatetimeIndex`: Datetime64[ns] precision data
# 6. `PeriodIndex`: Regularly-spaced, arbitrary precision datetime data.
# 7. `TimedeltaIndex`: Timedelta data
# 8. `CategoricalIndex`:
#
# Most of the time these are automatically created

# %% [markdown] Collapsed="false"
# ### Row slicing

# %% [markdown] Collapsed="false"
# User row slicing for returning subset

# %% Collapsed="false"
# Using indexes
weather.loc["DSM"].head()

# %% Collapsed="false"
# Not using indexes (harder)
weather2 = weather.reset_index()
weather2[weather2['station'] == 'DSM'].head()

# %% [markdown] Collapsed="false"
# ### Indexes for easier arithmetic, analysis

# %% [markdown] Collapsed="false"
# Translate from Fahrenheit to Celsius.

# %% Collapsed="false"
# Using indexes
temp = weather["tmpf"]
c = (temp - 32) * 5 / 9
c.to_frame()

# %% Collapsed="false"
# Not using indexes
temp2 = weather.reset_index()[["station", "date", "tmpf"]]
temp2["tmpf"] = (temp2["tmpf"] - 32) * 5 / 9
temp2.head()

# %% [markdown] Collapsed="false"
# ### Indexes for alignment

# %% [markdown] Collapsed="false"
# Let's create two DataFrame with different indexes

# %% [markdown] Collapsed="false"
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>.div</code><b> for dividing two series while controlling axis and fill value for NaN</li>
# </div>

# %% Collapsed="false"
dsm = weather.loc["DSM"]
hourly = dsm.resample("H").mean()
temp = hourly["tmpf"].sample(frac=0.5, random_state=1).sort_index()
sped = hourly["sped"].sample(frac=0.5, random_state=2).sort_index()
temp.head().to_frame()

# %% Collapsed="false"
sped.head()

# %% [markdown] Collapsed="false"
# Pandas will automatically align things due to index when performing operations

# %% Collapsed="false"
(sped / temp).to_frame()

# %% [markdown] Collapsed="false"
# Can also use `.div` to specify a fill value for the indexes that do not match

# %% Collapsed="false"
sped.div(temp, fill_value=1).to_frame()

# %% [markdown] Collapsed="false"
# Also can use `axis` parameter in `.div` to control alignment axis

# %% Collapsed="false"
hourly

# %% Collapsed="false"
hourly.div(sped, axis="index")

# %% Collapsed="false"
# Doing the same without index labels
# Messy.
temp2 = temp.reset_index()
sped2 = sped.reset_index()
# Find rows where the operation is defined
common_dates = pd.Index(temp2.date) & sped2.date
pd.concat(
    [
        # concat to not lose date information
        sped2.loc[sped2["date"].isin(common_dates), "date"],
        (
            sped2.loc[sped2.date.isin(common_dates), "sped"]
            / temp2.loc[temp2.date.isin(common_dates), "tmpf"]
        ),
    ],
    axis=1,
).dropna(how="all")

# %% [markdown] Collapsed="false"
# ## Merging

# %% [markdown] Collapsed="false"
# Two ways to merge:
#
# 1. `pd.merge`
# 2. `pd.concat`

# %% [markdown] Collapsed="false"
# ### Concat

# %% Collapsed="false"
pd.concat([temp, sped], axis=1).head()

# %% Collapsed="false"
pd.concat([temp, sped], axis=1, join="inner").head()

# %% [markdown] Collapsed="false"
# ### Merge version

# %% [markdown] Collapsed="false"
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>.agg</code><b> to pass different aggregate functions per column</li>
# </div>

# %% Collapsed="false"
pd.merge(temp.to_frame(), sped.to_frame(), left_index=True, right_index=True).head()

# %% Collapsed="false"
pd.merge(
    temp.to_frame(), sped.to_frame(), left_index=True, right_index=True, how="outer"
).head()

# %% [markdown] Collapsed="false"
# One to many joins. First will resample weather data into daily frequency by station.

# %% Collapsed="false"
idx_cols = ["unique_carrier", "origin", "dest", "tail_num", "fl_num", "fl_date"]
data_cols = [
    "crs_dep_time",
    "dep_delay",
    "crs_arr_time",
    "arr_delay",
    "taxi_out",
    "taxi_in",
    "wheels_off",
    "wheels_on",
]

df = flights.set_index(idx_cols)[data_cols].sort_index()


# %% Collapsed="false"
def mode(x):
    """
    Arbitrarily break ties.
    """
    return x.value_counts().index[0]


aggfuncs = {
    "tmpf": "mean",
    "relh": "mean",
    "sped": "mean",
    "mslp": "mean",
    "p01i": "mean",
    "vsby": "mean",
    "gust_mph": "mean",
    "skyc1": mode,
    "skyc2": mode,
    "skyc3": mode,
}
# Grouper works on a DatetimeIndex, so we move `station` to the
# columns and then groupby it as well.
daily = (
    weather.reset_index(level="station")
    .groupby([pd.Grouper(freq="1d"), "station"])
    .agg(aggfuncs)
)

daily.head()

# %% [markdown] Collapsed="false"
# ### Merge

# %% Collapsed="false"
daily_ = (
    daily.reset_index()
    .rename(columns={"date": "fl_date", "station": "origin"})
    .assign(
        origin=lambda _df: pd.Categorical(
            _df.origin, categories=flights.origin.cat.categories
        )
    )
)

# %% Collapsed="false"
daily_.head()

# %% Collapsed="false"
flights.head()

# %% Collapsed="false"
m = pd.merge(flights, daily_, on=["fl_date", "origin"]).set_index(idx_cols).sort_index()

m.head()

# %% [markdown] Collapsed="false"
# Well... that didn't work. `daily_` is from 2014 while `flights` is from 2017, so no surprise there for me. Not sure what references were going for...

# %% Collapsed="false"
