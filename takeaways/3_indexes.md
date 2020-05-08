# Indexes takeaways

## Random

`json_normalize` a good method for flattening json

## Slicing

- `.loc` for label-based indexing.
- `.iloc` for positional indexing.
- `pd.IndexSlice` for easy slicing of MultiIndexes

## Convenience index methods

- `MultiIndex.level` for grabbing specific indexes from a MultiIndex
- `Index.difference` for finding differences in indexes (like sets)

## Merging

Two ways:

1. `pd.merge`
2. `pd.concat`

## Grouping

- `.agg` to pass different aggregate functions per column
- `pd.Grouper` to resample and group by at the same time
