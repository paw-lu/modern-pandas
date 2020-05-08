# Performance takeaways

## Performance tips

- `pd.concat` on all DataFrame faster than `.append` each one at a time
- Don't use `.apply`, better to write functions that act on entire array.
  - Even looping though with `.itertuples` is faster than `.apply`.

## Converting data

- `to_numeric`
- `to_datetime`
- `to_timedelta`

## Other techniques

- `MultiIndex.from_product` to make indexes from product
- `.add_suffix` to add suffix to column names
- `.redindex`
- `.transform` to apply a function to a GroupBy and get values in original shape
