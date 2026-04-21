import qlib
from qlib.data import D
import pandas as pd

# 1. Initialize Qlib to point to your new data folder
provider_uri = "~/.qlib/qlib_data/us_data"
qlib.init(provider_uri=provider_uri, region="us")

# 2. Fetch the entire trading calendar available in the dataset
print("\n--- Fetching Dataset Calendar ---")
cal = D.calendar(freq='day')

# The calendar is an array of pandas Timestamps. We grab the first and last elements.
dataset_start = cal[0].strftime('%Y-%m-%d')
dataset_end = cal[-1].strftime('%Y-%m-%d')

print(f"Total Trading Days Available: {len(cal)}")
print(f"Dataset Date Range: {dataset_start} to {dataset_end}")

# 3. Use the calendar to dynamically define a date range for a specific stock
stock_ticker = "AAPL"
# Let's grab just the last 5 days of available data in the dataset
recent_start = cal[-5].strftime('%Y-%m-%d')
recent_end = dataset_end

fields = ['$open', '$high', '$low', '$close', '$volume']

print(f"\n--- Loading recent data for {stock_ticker} ({recent_start} to {recent_end}) ---")

# 4. Query the data
df = D.features([stock_ticker], fields, start_time=recent_start, end_time=recent_end)
print(df)