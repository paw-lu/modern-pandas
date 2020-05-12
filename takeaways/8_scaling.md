# Scaling takeaways

`nlargest` is an efficient way to get the top n values.

Dask is a library that creates computation graphs
to parallelize Python.

- `dask.datagrame.vizalize`
  lets you view the constructed computation graph
- `dask.dataframe.compute`
  fetches the data—
  Dask is lazy
- `dask.distributed.Client`
  is used to execute multiple graphs efficiently
  by reusing shared computations
- `dask.dataframe.merge` to merge Dask DataFrame
