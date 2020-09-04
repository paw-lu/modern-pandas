# Modern Pandas

A collection of best practices for using Pandas effectively. Each section includes a
Python and accompanying Markdown file that may be synced to a Jupyter Notebook via
[Jupytext](https://github.com/mwouts/jupytext)

_A lot_ of examples and code were collected from the excellent
[**datas-frame**](https://tomaugspurger.github.io/archives.html) blog.

## Table of contents

- [**Intro**](modern-pandas/intro.md)
  Load the data, and introduce `.loc`, `.iloc`,
  `SettingWithCopy` warning, and Multidimensional indexes.
- [**Method chaining**](modern-pandas/method_chaining.md)
  Chain methods together for more performant and idiomatic Pandas
- [**Indexes**](/modern-pandas/3_indexes.md)
  Effectively slice DataFrames
- [**Visualization**](modern-pandas/6_visualization.md)
  Leverage Seaborn and Pandas' built in plotting functionality
  to create charts

## References

- **[Why is Nobody Talking about Pandas NamedAgg?](https://deanla.com/pandas_named_agg.html)**
  Blog post on `NamedAgg`, and good uses of Pandas' `.agg`
