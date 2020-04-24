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
# # Modern Pandas

# %% [markdown] Collapsed="false"
# ## Get data

# %% [markdown] Collapsed="false"
# Just downloading data here. Feel free to ignore 😅.

# %% Collapsed="false"
import os
import zipfile

import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% Collapsed="false"
headers = {
    "Referer": "https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time",
    "Origin": "https://www.transtats.bts.gov",
    "Content-Type": "application/x-www-form-urlencoded",
}

params = (
    ("Table_ID", "236"),
    ("Has_Group", "3"),
    ("Is_Zipped", "0"),
)

with open("modern-1-url.txt", encoding="utf-8") as f:
    data = f.read().strip()

os.makedirs("data", exist_ok=True)
dest = "data/flights.csv.zip"

if not os.path.exists(dest):
    r = requests.post(
        "https://www.transtats.bts.gov/DownLoad_Table.asp",
        headers=headers,
        params=params,
        data=data,
        stream=True,
    )

    with open("data/flights.csv.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=102400):
            if chunk:
                f.write(chunk)

# %% Collapsed="false"
zf = zipfile.ZipFile("data/flights.csv.zip")
fp = zf.extract(zf.filelist[0].filename, path="data/")
df = pd.read_csv(fp, parse_dates=["FL_DATE"]).rename(columns=str.lower)

df.info()

# %% [markdown] Collapsed="false"
# ## Index

# %% [markdown] Collapsed="false"
# Two methods to get rows:
#
# 1. Use `.loc` for label-based indexing
# 2. Use `.iloc` for positional indexing

# %% Collapsed="false"
first = df.groupby("unique_carrier").first()

# %% Collapsed="false"
first.loc[["AA", "AS", "DL"], ["fl_date", "tail_num"]]

# %% Collapsed="false"
first.iloc[[0, 1, 3], [0, 1]]

# %% [markdown] Collapsed="false"
# ## SettingWithCopy

# %% [markdown] Collapsed="false"
# Do not let the ends of two square brackets touch `][`. This does _not_ result in an an assignment to column `"b"`:
#
# ```python
# # This is bad, do not do
# f[f["a"] <= 3]["b"] = f[f["a"] <= 3]["b"] / 10
# ```

# %% Collapsed="false"
# Correct way
f = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
f.loc[f["a"] <= 3, "b"] = f.loc[f["a"] <= 3, "b"] / 10
f

# %% [markdown] Collapsed="false"
# ## Multidimensional indexing

# %% Collapsed="false"
hdf = df.set_index(
    ["unique_carrier", "origin", "dest", "tail_num", "fl_date"]
).sort_index()
hdf[hdf.columns[:4]].head()

# %% [markdown] Collapsed="false"
# Selecting outermost index

# %% Collapsed="false"
hdf.loc[["AA", "DL", "US"], ["dep_time", "dep_delay"]]

# %% [markdown] Collapsed="false"
# Selecting first two using a tuple `()`.

# %% Collapsed="false"
hdf.loc[(["AA", "DL", "US"], ["ORD", "DSM"]), ["dep_time", "dep_delay"]]

# %% [markdown] Collapsed="false"
# Selecting only second index using `pd.IndexSlice`.

# %% Collapsed="false"
hdf.loc[pd.IndexSlice[:, ["ORD", "DSM"]], ["dep_time", "dep_delay"]]

# %% Collapsed="false"
pd.IndexSlice[:, ['ORD', 'DSM']]

# %% Collapsed="false"
