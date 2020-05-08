# Visualization takeaways

## Matplotlib

Can take in labeled data directly.

```python
fig, ax = plt.subplots()

ax.scatter(x="carat", y="depth", data=df, c="k", alpha=0.15)
```

## Pandas

```python
df.plot.scatter(x="carat", y-"depth", c="k", alpha=0.15)
```

## Seaborn

- `FacetGrid` for grid of plots with different values
- `PairGriid` for grid of plots with different pairs

## Other

pandas_datareader library is able to pull data from various sources.

- `.quantile` to get value of certain quantile
- `.all` to see if all values in column are True
