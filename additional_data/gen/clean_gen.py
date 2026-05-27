import pandas as pd
import os

CURRENT_DIR = current_dir = os.path.dirname(os.path.abspath(__file__))

def prepare(csv_name, output_name, prefix):
    df_prices = load_raw_data(csv_name)

    df = clean_data(df_prices)
    
    df = handle_datetime_issues(df)

    df = create_lag_features(df, prefix)
    
    df = create_rolling_features(df, prefix)
    
    df.to_csv(os.path.join(CURRENT_DIR, output_name), index=False)
    df = df.drop(['value'], axis=1)
    df = df[df['utc'].between('2023-01-15 00:00:00', '2025-10-05 19:00:00')]
    return df

def clean_generation():
    df = prepare("generation.csv","generation_cleaned.csv", "generation")
    df.to_csv(os.path.join(CURRENT_DIR, "generation_cleaned.csv"), index=False)
    return df
    
def create_lag_features(df, prefix, lag_hours=[1, 2, 3, 6, 12, 24, 48, 96, 168]):
    for lag in lag_hours:
        df[f'{prefix}_{lag}'] = df['value'].shift(lag)

    df = remove_nan_rows(df)
    return df

def create_rolling_features(df,prefix, windows=[6, 12, 24, 48, 168]):
    shifted = df['value'].shift(1)
    for window in windows:
        df[f'{prefix}_rolling_mean_{window}'] = shifted.rolling(window=window).mean()
        df[f'{prefix}_rolling_std_{window}'] = shifted.rolling(window=window).std()

    df = remove_nan_rows(df)
    return df

def remove_nan_rows(df):    
    df_ml = df.dropna()
    return df_ml

def load_raw_data(filepath):
    df = pd.read_csv(os.path.join(CURRENT_DIR, filepath))
    return df

def clean_data(df):
    if(df.isna().sum().sum() > 0):
        df = df.dropna()

    df_clean = df.drop(['ltu', 'id'], axis=1)
    return df_clean

def handle_datetime_issues(df_clean):
    df_clean['utc'] = pd.to_datetime(df_clean['utc'])
    
    duplicates = df_clean['utc'].duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates(subset=['utc'], keep='first')
        
    min_date = df_clean['utc'].min()
    max_date = df_clean['utc'].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq='h')

    df_clean = df_clean.set_index('utc')
    df_clean = df_clean.reindex(full_range)
    
    df_clean = df_clean.interpolate(method='linear')
    df_clean = df_clean.reset_index().rename(columns={'index': 'utc'})
    
    return df_clean