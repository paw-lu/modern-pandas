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
# # Tidy data

# %% [markdown] Collapsed="false"
# Tidy data is:
#
# 1. Each variable forms a column
# 2. Each observation forms a row
# 3. Each type of observational unit forms a table

# %% [markdown] Collapsed="false"
# ## Imports

# %% Collapsed="false"
# %matplotlib inline

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_rows = 10
sns.set(style='ticks', context='talk')

# %% [markdown] Collapsed="false"
# ## Data

# %% [markdown] Collapsed="false"
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>.read_html</code></b> to read html tables.</li>
#     <li><b><code>.dropna</code></b> is well known, but good to know that it has a `thresh` argument that let's you set how many items should be missing per row.</li>
#     <li><b><code>.read_html</code></b> to read html tables.</li>
#     <li><b><code>.assign</code></b> can take callable (<code>lambda</code>)</li>
#     <li><b><code>.set_index</code></b> has an <code>append</code> keyword to add new index to existing one</li>
#     <li><b><code>.rename_axis</code></b> can just take in a list of names</li>
# </div>

# %% Collapsed="false"
fp = "data/nba.csv"

if not os.path.exists(fp):
    tables = pd.read_html(
        "http://www.basketball-reference.com/leagues/NBA_2016_games.html"
    )
    games = tables[0]
    games.to_csv(fp)
else:
    games = pd.read_csv(fp)
games.head()

# %% [markdown] Collapsed="false"
# Clean data

# %% Collapsed="false"
column_names = {
    "Date": "date",
    "Start (ET)": "start",
    "Unamed: 2": "box",
    "Visitor/Neutral": "away_team",
    "PTS": "away_points",
    "Home/Neutral": "home_team",
    "PTS.1": "home_points",
    "Unamed: 7": "n_ot",
}
games = (
    games.rename(columns=column_names)
    .dropna(thresh=4)[["date", "away_team", "away_points", "home_team", "home_points"]]
    .assign(date=lambda x: pd.to_datetime(x["date"], format="%a, %b %d, %Y"))
    .set_index("date", append=True)
    .rename_axis(["game_id", "date"])
    .sort_index()
)
games.head()

# %% [markdown] Collapsed="false"
# ## Make tidy

# %% [markdown] Collapsed="false"
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>pd.melt</code></b> to unpivot table (wide to long)</li>
#     <li><b><code>.diff</code></b> to find difference between this column and last one. Can use with <code>.groupby</code></li>
#     <li><b><code>pd.pivot_table</code></b> to pivot DataFrame (long to wide)</li>
# </div>

# %% [markdown] Collapsed="false"
# **How many days of rest did each team get in between each game?**

# %% Collapsed="false"
games.head()

# %% Collapsed="false"
tidy = pd.melt(
    games.reset_index(),
    id_vars=["game_id", "date"],
    value_vars=["away_team", "home_team"],
    value_name="team",
)
tidy.head()

# %% [markdown] Collapsed="false"
# Now get number of days between games

# %% Collapsed="false"
tidy["rest"] = tidy.sort_values("date").groupby("team")["date"].diff().dt.days - 1
tidy.dropna().head()

# %% [markdown] Collapsed="false"
# Return to pivoted table

# %% Collapsed="false"
by_game = pd.pivot_table(
    tidy, values="rest", index=["game_id", "date"], columns="variable"
).rename(columns={"away_team": "away_rest", "home_team": "home_rest"})
df = pd.concat([games, by_game], axis=1)
df.dropna().head()

# %% [markdown] Collapsed="false"
# **What was each team's average days of rest, at home and on the road?**

# %% Collapsed="false"
tidy.head()

# %% Collapsed="false"
sns.set(style="ticks", context="paper")
g = sns.FacetGrid(tidy, col="team", col_wrap=6, hue="team", size=2)
g.map(sns.barplot, "variable", "rest");

# %% [markdown] Collapsed="false"
# **Distribution of rest differences in games**

# %% Collapsed="false"
df["home_win"] = df["home_points"] > df["away_points"]
df["rest_spread"] = df["home_rest"] - df["away_rest"]
df.dropna().head()

# %% Collapsed="false"
delta = (by_game.home_rest - by_game.away_rest).dropna().astype(int)
ax = (
    delta.value_counts()
    .reindex(np.arange(delta.min(), delta.max() + 1), fill_value=0)
    .sort_index()
    .plot(kind="bar", color="k", width=0.9, rot=0, figsize=(12, 6))
)
sns.despine()
ax.set(xlabel="Difference in Rest (Home - Away)", ylabel="Games");

# %% [markdown] Collapsed="false"
# Win percentage by rest difference

# %% Collapsed="false"
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x="rest_spread",
    y="home_win",
    data=df.query("-3 <= rest_spread <= 3"),
    color="#4c72b0",
    ax=ax,
)
sns.despine()

# %% [markdown] Collapsed="false"
# ## Stack / Unstack

# %% [markdown] Collapsed="false"
# <div class="alert alert-block alert-info">
#     <b>Techniques of note</b>
#     <br><br>
#     <li><b><code>.stack</code></b> convert from wide to long format</li>
#     <li><b><code>.unstack</code></b> convert from long to wide format</li>
# </div>

# %% Collapsed="false"
rest = tidy.groupby(["date", "variable"]).rest.mean().dropna()
rest.head()

# %% [markdown] Collapsed="false"
# `rest` is now in "long" form.

# %% Collapsed="false"
rest.unstack().head()

# %% Collapsed="false"
rest.unstack().stack()

# %% [markdown] Collapsed="false"
# `DataFrame.plot()` expects wide-form data—one line per column

# %% Collapsed="false"
with sns.color_palette() as pal:
    b, g = pal.as_hex()[:2]

ax = (
    rest.unstack()
    .query("away_team < 7")
    .rolling(7)
    .mean()
    .plot(figsize=(12, 6), linewidth=3, legend=False)
)
ax.set(ylabel="Rest (7 day MA)")
ax.annotate("Home", (rest.index[-1][0], 1.02), color=g, size=14)
ax.annotate("Away", (rest.index[-1][0], 0.82), color=b, size=14)
sns.despine()

# %% [markdown] Collapsed="false"
# ## Home court advantage

# %% [markdown] Collapsed="false"
# ### 1. Create outcome variable

# %% Collapsed="false"
df["home_win"] = df.away_points < df.home_points

# %% [markdown] Collapsed="false"
# ### 2. Find win % per team

# %% Collapsed="false"
df.head()

# %% Collapsed="false"
wins = (
    pd.melt(
        df.reset_index(),
        id_vars=["game_id", "date", "home_win"],
        value_name="team",
        var_name="is_home",
        value_vars=["home_team", "away_team"],
    )
    .assign(win=lambda x: x.home_win == (x.is_home == "home_team"))
    .groupby(["team", "is_home"])
    .win.agg(["sum", "count", "mean"])
    .rename(columns=dict(sum="n_wins", count="n_games", mean="win_pct"))
)
wins.head()

# %% Collapsed="false"
g = sns.FacetGrid(wins.reset_index(), hue="team", size=7, aspect=0.5, palette=["k"])
g.map(sns.pointplot, "is_home", "win_pct").set(ylim=(0, 1));

# %% Collapsed="false"
g = sns.FacetGrid(wins.reset_index(), col="team", hue="team", col_wrap=5, size=2)
g.map(sns.pointplot, "is_home", "win_pct")

# %% Collapsed="false"
win_percent = (
    # Use sum(games) / sum(games) instead of mean
    # since I don't know if teams play the same
    # number of games at home as away
    wins.groupby(level="team", as_index=True).apply(
        lambda x: x.n_wins.sum() / x.n_games.sum()
    )
)
win_percent.head()

# %% Collapsed="false"
win_percent.sort_values().plot.barh(figsize=(6, 12), width=0.85, color="k")
plt.tight_layout()
sns.despine()
plt.xlabel("Win Percent")

# %% Collapsed="false"
plt.figure(figsize=(8, 5))
(
    wins.win_pct.unstack()
    .assign(
        **{
            "Home Win % - Away %": lambda x: x.home_team - x.away_team,
            "Overall %": lambda x: (x.home_team + x.away_team) / 2,
        }
    )
    .pipe((sns.regplot, "data"), x="Overall %", y="Home Win % - Away %")
)
sns.despine()
plt.tight_layout()

# %% Collapsed="false"
df = df.assign(
    away_strength=df["away_team"].map(win_percent),
    home_strength=df["home_team"].map(win_percent),
    point_diff=df["home_points"] - df["away_points"],
    rest_diff=df["home_rest"] - df["away_rest"],
)
df.head()

# %%
