import pandas as pd
import os

def load_raw_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded data from {filepath} with shape {df.shape}")
    print(f"Head:\n{df.head()}")
    return df

def clean_data(df):
    total_nas = df.isna().sum().sum()
    print(f"Total N/A values: {total_nas}")
    if(df.isna().sum().sum() > 0):
        print("Dropping rows with N/A values...")
        df = df.dropna()
        print(f"Data shape after dropping N/A: {df.shape}")

    df_clean = df.drop(['ltu', 'id'], axis=1)
    print("\nCLEANED DATA (ltu and id removed):")
    print(df_clean.head(10))
    print("\n")
    return df_clean

def handle_datetime_issues(df_clean):
    df_clean['utc'] = pd.to_datetime(df_clean['utc'])
    
    duplicates = df_clean['utc'].duplicated().sum()
    print(f"Number of duplicate timestamps: {duplicates}")
    
    if duplicates > 0:
        print("\n" + "=" * 60)
        print("DUPLICATE TIMESTAMPS:")
        print("=" * 60)
        duplicate_timestamps = df_clean[df_clean['utc'].duplicated(keep=False)]['utc'].unique()
        print("The following timestamps have duplicates:")
        for timestamp in sorted(duplicate_timestamps):
            print(f"  {timestamp}")
    
    min_date = df_clean['utc'].min()
    max_date = df_clean['utc'].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq='h')
    missing = full_range.difference(df_clean['utc'])
    
    print(f"Total expected hours: {len(full_range)}")
    print(f"Missing hours: {len(missing)}")
    print(missing)
    
    df_clean = df_clean.set_index('utc')
    df_clean = df_clean.reindex(full_range)
    print(f"NaNs before linear interpolation: {df_clean.isnull().sum().sum()}")
    
    df_clean = df_clean.interpolate(method='linear')
    df_clean = df_clean.reset_index().rename(columns={'index': 'utc'})
    print(f"NaNs after linear interpolation: {df_clean.isnull().sum().sum()}")
    
    return df_clean