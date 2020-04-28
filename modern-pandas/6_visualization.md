---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.0
  kernelspec:
    display_name: modern-pandas
    language: python
    name: modern-pandas
---

<!-- #region Collapsed="false" -->
# Visualization and exploratory analysis
<!-- #endregion -->

```python Collapsed="false"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

<!-- #region Collapsed="false" -->
## Data
<!-- #endregion -->

```python Collapsed="false"
df = pd.read_csv(
    "http://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/diamonds.csv",
    index_col=0,
)
df.head()
```

```python Collapsed="false"
df.info()
```

```python Collapsed="false"
sns.set(context="talk", style="ticks")

%matplotlib inline
```

<!-- #region Collapsed="false" -->
## Matplotlib
<!-- #endregion -->

<!-- #region Collapsed="false" -->
Matplotlib can plot labeled data now I guess
<!-- #endregion -->

```python Collapsed="false"
fig, ax = plt.subplots()

ax.scatter(x="carat", y="depth", data=df, c="k", alpha=0.15)
```

<!-- #region Collapsed="false" -->
## Pandas built-int
<!-- #endregion -->

<!-- #region Collapsed="false" -->
<div class="alert alert-block alert-info">
    <H3>Techniques of note</H3>
    <li><b><code>pandas_datareader</code></b> library to pull data from various sources into a DataFrame</li>
</div>
<!-- #endregion -->

```python Collapsed="false"
df.plot.scatter(x="carat", y="depth", c="k", alpha=0.15)
plt.tight_layout()
```

<!-- #region Collapsed="false" -->
Convenient to plot in when index is timeseries
<!-- #endregion -->

```python Collapsed="false"
from pandas_datareader import fred

gdp = fred.FredReader(["GCEC96", "GPDIC96"], start="2000-01-01").read()
gdp.rename(
    columns={"GCEC96": "Government Expenditure", "GPDIC96": "Private Investment"}
).plot(figsize=(12, 6))
plt.tight_layout()
```

```python Collapsed="false"
gdp.head()
```

<!-- #region Collapsed="false" -->
## Seaborn
<!-- #endregion -->

<!-- #region Collapsed="false" -->
<div class="alert alert-block alert-info">
    <h3>Techniques of note</h3>
    <h4>Pandas</h4>
    <li><b><code>.quantile</code></b> get the value of a certain quantile per column</li>
    <li><b><code>.all</code></b> to test if all values in a column are True</li>
    <h4>Seaborn</h4>
    <li><b><code>countplot</code></b> histogram</li>
    <li><b><code>jointplot</code></b> scatter</li>
    <li><b><code>pairplot</code></b> pairwise scatter and histogram</li>
    <li><b><code>PairGrid</code></b> to make a seaborn grid of pairs</li>
    <li><b><code>FacetGrid</code></b> to make a seaborn grid of different values</li>
    <li><b><code>catplot</code></b> to make a seaborn out of different categories</li>
    <li><b><code>.map</code></b> For assigning different plots seaborn <code>Grid</code>s</li>
    <li><b><code>.map_upper/.map_diag/.map_lower</code></b> For assigning different plots to <code>PairGrid</code></li>
    
</div>
<!-- #endregion -->

```python Collapsed="false"
sns.countplot(x="cut", data=df)
sns.despine()
plt.tight_layout()
```

```python Collapsed="false"
sns.jointplot(x="carat", y="price", data=df, height=8, alpha=0.25, color="k", marker=".")
plt.tight_layout()
```

```python Collapsed="false"
g = sns.pairplot(df, hue="cut")
```

<!-- #region Collapsed="false" -->
Seaborn has `Grid`s—which you can use to plot functions over each axis.
<!-- #endregion -->

```python Collapsed="false"
def core(df, α=0.05):
    mask = (df > df.quantile(α)).all(1) & (df < df.quantile(1 - α)).all(1)
    return df[mask]
```

```python Collapsed="false"
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

(
    df.select_dtypes(include=[np.number])
    .pipe(core)
    .pipe(sns.PairGrid)
    .map_upper(plt.scatter, marker=".", alpha=0.25)
    .map_diag(sns.kdeplot)
    .map_lower(plt.hexbin, cmap=cmap, gridsize=20)
);
```

```python Collapsed="false"
agged = df.groupby(["cut", "color"]).mean().sort_index().reset_index()

g = sns.PairGrid(
    agged, x_vars=agged.columns[2:], y_vars=["cut", "color"], size=5, aspect=0.65
)
g.map(sns.stripplot, orient="h", size=10, palette="Blues_d");
```

```python Collapsed="false"
g = sns.FacetGrid(df, col="color", hue="color", col_wrap=4)
g.map(sns.regplot, "carat", "price");
```

```python Collapsed="false"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

```python Collapsed="false"
df = sns.load_dataset("titanic")

clf = RandomForestClassifier()
param_grid = dict(
    max_depth=[1, 2, 5, 10, 20, 30, 40],
    min_samples_split=[2, 5, 10],
    min_samples_leaf=[2, 3, 5],
)
est = GridSearchCV(clf, param_grid=param_grid, n_jobs=4)

y = df["survived"]
X = df.drop(["survived", "who", "alive"], axis=1)

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(value=X.median())
est.fit(X, y);
```

```python Collapsed="false"
scores = pd.DataFrame(est.cv_results_)
scores.head()
```

```python Collapsed="false"
sns.catplot(
    x="param_max_depth",
    y="mean_test_score",
    col="param_min_samples_split",
    hue="param_min_samples_leaf",
    data=scores,
);
```

```python Collapsed="false"

```
