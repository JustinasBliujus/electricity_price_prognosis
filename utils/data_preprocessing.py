import pandas as pd

def load_raw_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    total_nas = df.isna().sum().sum()
    print(f"total nan values: {total_nas}")
    if(df.isna().sum().sum() > 0):
        df = df.dropna()
        print(f"after dropping nan: {df.shape}")

    df_clean = df.drop(['ltu', 'id'], axis=1)
    print("\ncleaned data (ltu and id removed):")
    print(df_clean.head(10))
    print("\n")
    return df_clean

def handle_datetime_issues(df_clean):
    df_clean['utc'] = pd.to_datetime(df_clean['utc'])
    
    duplicates = df_clean['utc'].duplicated().sum()
    print(f"number of duplicate timestamps: {duplicates}")
    
    if duplicates > 0:
        print("duplicate timestamps:")
        duplicate_timestamps = df_clean[df_clean['utc'].duplicated(keep=False)]['utc'].unique()
        print("duplicates:")
        for timestamp in sorted(duplicate_timestamps):
            print(f"  {timestamp}")
    
    min_date = df_clean['utc'].min()
    max_date = df_clean['utc'].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq='h')
    missing = full_range.difference(df_clean['utc'])
    
    print(f"total expected hours: {len(full_range)}")
    print(f"missing hours: {len(missing)}")
    print(missing)
    
    df_clean = df_clean.set_index('utc')
    df_clean = df_clean.reindex(full_range)
    print(f"nans before linear interpolation: {df_clean.isnull().sum().sum()}")
    
    df_clean = df_clean.interpolate(method='linear')
    df_clean = df_clean.reset_index().rename(columns={'index': 'utc'})
    print(f"nans after linear interpolation: {df_clean.isnull().sum().sum()}")
    
    return df_clean