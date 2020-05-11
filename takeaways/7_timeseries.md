# Timeseries takeaways

## Special slicing

- `df.loc["2016"]` slice timestamp indexes by year.

## Special methods

- `.resample` group by time frame
- `.rolling` group by number of rows
- `.expanding` apply function to rows before
- `.ewm` Apply exponentially weighted functions

## Grab bad

- `DateOffset` to add offsets to dates
- `tseries.holiday` to get holidays
- `tz_localize` to assign a timezone
- `tz_convert` to convert to different timezone

## Time series modeling

- `.shift` to add lagged values
- `smt.SARIMAX` for ARIMA time series modeling
