import os
import torch
import pandas as pd
import qlib
from qlib.tests.data import GetData
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import numpy as np # Ensure this is at the top of your file

def initialize_qlib():
    """Initializes Qlib and auto-downloads the data if it's missing (Crucial for Colab)."""
    provider_uri = os.path.expanduser("~/.qlib/qlib_data/us_data")
    
    # The crucial Colab check I accidentally removed earlier!
    if not os.path.exists(os.path.join(provider_uri, "instruments")):
        print(f"Qlib data not found at {provider_uri}. Downloading now...")
        GetData().qlib_data(target_dir=provider_uri, region="us")
        print("Download complete!")
    
    qlib.init(provider_uri=provider_uri, region="us")

def get_qlib_dataloader(start_date, end_date, seq_len=20):
    """
    Generates Alpha158 features on-the-fly using a sliding window 
    to prevent Out-Of-Memory (OOM) crashes on large datasets.
    """
    print(f"Fetching Alpha158 features for {start_date} to {end_date}...")
    
    handler = Alpha158(instruments="all", start_time=start_date, end_time=end_date)
    dataset = DatasetH(handler=handler, segments={"data": (start_date, end_date)})
    
    # Fetch features and labels (Note: data_key argument is omitted to prevent KeyErrors)
    df = dataset.prepare("data", col_set=["feature", "label"])
    
    # Drop rows with NaN values
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(df.head())
    
    # Get a sorted list of all unique trading days
    dates = df.index.get_level_values('datetime').unique().sort_values()
    
    # Slide a 20-day window across the timeline
    for i in range(seq_len - 1, len(dates)):
        target_date = dates[i]
        window_dates = dates[i - seq_len + 1 : i + 1]
        
        # Extract just the 20 days of data
        window_df = df.loc[window_dates]
        
        # Keep ONLY stocks that have data for all 20 consecutive days
        stock_counts = window_df.index.get_level_values('instrument').value_counts()
        valid_stocks = stock_counts[stock_counts == seq_len].index
        
        if len(valid_stocks) < 20:  # NUM_PORTFOLIOS minimum
            continue
            
        # Filter the window to our valid stocks and sort to ensure alignment
        valid_window = window_df[window_df.index.get_level_values('instrument').isin(valid_stocks)]
        valid_window = valid_window.sort_index(level=['instrument', 'datetime'])
        
        # Reshape features to (N_stocks, 20_days, 158_features)
        features = valid_window['feature'].values.reshape(len(valid_stocks), seq_len, -1)
        
        # The labels (returns) should only be from the target_date (today)
        labels = valid_window.xs(target_date, level='datetime')['label'].values.flatten()
        
        x_batch = torch.tensor(features, dtype=torch.float32)
        y_batch = torch.tensor(labels, dtype=torch.float32)
        
        # Yield streams the data efficiently directly to the GPU without bloating RAM
        yield target_date, x_batch, y_batch