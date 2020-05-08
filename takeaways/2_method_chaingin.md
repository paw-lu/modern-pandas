# Method chaining takeaways

## Chain-friendly methods

- `.assign`
- `.pipe`
- `.remane`/`.rename_axis`
- `.rolling`
- `resample`
- `.where`
- `.mask`
- `.stack`
- `.unstack`
- `pd.Grouper`

## Using decorators to inspect mid-chain

```python
from functools import wraps
import logging

def log_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.info("%s,%s" % (func.__name__, result.shape))
        return result
    return wrapper

def log_dtypes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.info("%s,%s" % (func.__name__, result.dtypes))
        return result
    return wrapper
```

## Using callables (`lambda`)

```python
df = (
  df.loc[lambda _df: _df["column"] == 7]
  .assign(
    big_column=lambda: _df["column"] + 1
  )[
    lambda _df: 0 < _df["big_column"]
  ]
)
```
