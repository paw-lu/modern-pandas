# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: modern-pandas
#     language: python
#     name: modern-pandas
# ---

# %% [markdown]
# # Scaling

# %% [markdown]
# Pandas DataFrames need to fit in RAM.
# You have two solutions for larger datasets:
#
# 1. Don't use Pandas
# 2. Iteration

# %% [markdown]
# ## Task

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>.nlargest</code></b> to efficiently find the top n values</li>
# </div>

# %% [markdown]
# Task is to find 100 most-common occupations.

# %%
from pathlib import Path

import pandas as pd
import seaborn as sns

df = pd.read_parquet("data/indiv-10.parq", columns=["occupation"], engine="pyarrow")

most_common = df.occupation.value_counts().nlargest(100)
most_common

# %% [markdown]
# Could only do for the year 2010,
# because data was too big.

# %% [markdown]
# ### Using iteration

# %% [markdown]
# To do for iteration,
# we need to rewrite the code
# in such a say that too much memory is not used at once.
#
# 1. Create a global `total_counts` Series
#    that contains the counts from all of the files processed thus far
# 2. Read in a file
# 3. Compute a temporary variable `counts`
#    with the counts for just one file
# 4. Add temporary `counts` to `total_counts`
# 5. Select the 100 largest

# %%
files = sorted(Path("data/").glob("indiv-*.parq"))

total_counts = pd.Series()

for year in files:
    df = pd.read_parquet(year, columns=["occupation"], engine="pyarrow")
    counts = df.occupation.value_counts()
    total_counts = total_counts.add(counts, fill_value=0)

total_counts = total_counts.nlargest(100).sort_values(ascending=False)

# %% [markdown]
# ### Using dask

# %%
import dask.dataframe as dd

df = dd.read_parquet("data/indiv-*.parquet", engine="pyarrow", columns=["occupation"])

most_common = df.occupation.value_counts().nlargest(100)
most_common.compute().sort_values(ascending=False)

# %% [markdown]
# ## Dask

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>dask.dataframe.visualize</code></b> to view computation graphs</li>
#     <li><b><code>dask.dataframe.compute</code></b> to fetch data</li>
#     <li><b><code>dask.distributed.Client</code></b> to execute multiple graphs efficiently—resusing shared computations</li>
# </div>

# %% [markdown]
# Dask parallizes Python.
# It provides APIs for
#
# - Arrays
# - DataFrames
# - Parallelizing custom algorithms
#
# Dask works with **task graphs**—
# functions to call on data
# and the relationships between tasks.
#
# Dash DataFrame consists of many pandas DataFrames arranged by index.
# Dash really just coordinates these pandas DataFrames.
#
# Dash DataFrames are lazy.

# %%
# Visualize graphs
df.visualize(rankdir="LR")

# %% [markdown]
# Can call methods to add tasks to graph.

# %%
most_common = df.occupation.value_counts().nlargest(100)
most_common

# %%
most_common.visualize(rankdir="LR")

# %% [markdown]
# `most_common` does not hold the answer,
# it holds a recipe for the answer—
# a list of steps to take.
#
# One way to get an answer out is to call `.compute`

# %%
most_common.compute()

# %% [markdown]
# At this point
# the graph is handed over to a scheduler.
#
# `dask.dataframe` and `dask.array`
# provide familiar APIs for working on large datasets
# Computations are represented as a task graph.
# Dask schedulers run task graphs in parallel.

# %%
import dask.dataframe as dd
import seaborn as sns
from dask import compute
from dask.distributed import Client

client = Client(processes=False)

# %% [markdown]
# Calling `Client`
# without providing a scheduler
# will make a local cluster of threads on your machine.

# %%
individual_cols = [
    "cmte_id",
    "entity_tp",
    "employer",
    "occupation",
    "transaction_dt",
    "transaction_amt",
]

indiv = dd.read_parquet("data/indiv-*.parq", columns=individual_cols, engine="pyarrow")
indiv

# %%
avg_transaction = indiv.transaction_amt.mean()

# %% [markdown]
# **Which employer's employees donated the most?**

# %%
total_by_employee = indiv.groupby("employer").transaction_amt.sum().nlargest(10)

# %% [markdown]
# **Or "what is the average amount donated per occupation?"**

# %%
avg_by_occupation = indiv.groupby("occupation").transaction_amt.mean().nlargest(10)

# %% [markdown]
# Again,
# Dask is lazy
# and has yet to compute anything.
#
# The three graphs created—
# `avg_transaction`,
# `total_by_employee`,
# and `avg_by_occupation`—
# have different task graphs
# but share some common structure.
# Dask is ample to avoid redundant calculations
# when `dask.compute` is used.

# %%
# %%time
avg_transaction, by_employee, by_occupation = compute(
    avg_transaction, total_by_employee, avg_by_occupation
)

# %% [markdown]
# Use filtering
# and find the 10 most common occupations.

# %%
top_occupations = (indiv.occupation.value_counts().nlargest(10).index).compute()
donations = (
    indiv[indiv.occupation.isin(top_occupations)]
    .groupby("occupation")
    .transaction_amt.agg(["count", "mean", "sum", "max"])
)

# %%
ax = occupation_avg.sort_values(ascending=False).plot.barh(color="k", width=0.9)
lim = ax.get_ylim()
ax.vlines(total_avg, *lim, color="C1", linewidth=3)
ax.legend(["Average donation"])
ax.set(xlabel="Donation Amount", title="Average Dontation by Occupation")
sns.despine()

# %% [markdown]
# Dask has Pandas' time-series support
# with methods like `resample`.

# %%
daily = (
    indiv[["transaction_dt", "transaction_amt"]]
    .dropna()
    .set_index("transaction_dt")["transaction_amt"]
    .resample("D")
    .sum()
).compute()
daily

# %% [markdown]
# Now filter out to just 2011–2016.
# Dash transitions seamlessly to Pandas operations.

# %%
subset = daily.loc["2011":"2016"]
ax = subset.div(1000).plot(figsize=(12, 6))
ax.set(
    ylim=0, title="Daily Donations", ylabel="$ (thousands)",
)
sns.despine()
;

# %% [markdown]
# ## Joining

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>dask.dataframe.merge</code></b> To merge Dash DataFrame</li>
# </div>

# %%
committee_cols = ["cmte_id", "cmte_nm", "cmte_tp", "cmte_pty_affiliation"]
cm = dd.read_parquet("data/cm-*.parq", columns=committee_cols).compute()

# Some committees change thier name, but the ID stays the same
cm = cm.groupby("cmte_id").last()
cm

# %%
indiv = indiv[
    (indiv.transaction_dt >= pd.Timestamp("2007-01-01"))
    & (indiv.transaction_dt <= pd.Timestamp("2018-01-01"))
]

df2 = dd.merge(indiv, cm.reset_index(), on="cmte_id")
df2

# %%
indiv = indiv.repartition(npartitions=10)
df2 = dd.merge(indiv, cm.reset_index(), on="cmte_id")
df2

# %%
party_donations = (
    (df2.groupby([df2.transaction_dt, "cmte_pty_affiliation"]).transaction_amt.sum())
    .compute()
    .sort_index()
)

# %%
ax = (
    party_donations.loc[:, ["REP", "DEM"]]
    .unstack("cmte_pty_affiliation")
    .iloc[1:-2]
    .rolling("30D")
    .mean()
    .plot(color=["C0", "C3"], figsize=(12, 6), linewidth=3)
)
sns.despine()
ax.set(title="Daily Donations (30-D Moving Average)", xlabel="Date")
;

# %% [markdown]
# Install by running
#
# ```sh
# pip install dask[complete]
# ```

# %%
