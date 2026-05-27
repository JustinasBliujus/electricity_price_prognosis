import pandas as pd
import numpy as np
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
    
def clean_all():
    LT_LV = prepare("LT_LV.csv","LT_LV_cleaned.csv", "LT_LV")
    LV_LT = prepare("LV_LT.csv","LV_LT_cleaned.csv", "LV_LT")
    LT_PL = prepare("LT_PL.csv","LT_PL_cleaned.csv", "LT_PL")
    PL_LT = prepare("PL_LT.csv","PL_LT_cleaned.csv", "PL_LT")
    LT_SW = prepare("LT_SW.csv","LT_SW_cleaned.csv", "LT_SW")
    SW_LT = prepare("SW_LT.csv","SW_LT_cleaned.csv", "SW_LT")
    
    print(f"LT_LV len {len(LT_LV)}")
    print(f"LV_LT len {len(LV_LT)}")
    print(f"LT_PL len {len(LT_PL)}")
    print(f"PL_LT len {len(PL_LT)}")
    print(f"LT_SW len {len(LT_SW)}")
    print(f"SW_LT len {len(SW_LT)}")
    
    dfs = [LT_LV, LV_LT, LT_PL, PL_LT, LT_SW, SW_LT]
    
    df = dfs[0]
    for right in dfs[1:]:
        df = df.merge(right, on='utc', how='inner')

    new_cols = {}

    for lag in [1,2,3,6,12,24,48,96,168]:
        new_cols[f'net_LT_LV_{lag}'] = df[f'LT_LV_{lag}'] - df[f'LV_LT_{lag}']
        new_cols[f'net_LT_PL_{lag}'] = df[f'LT_PL_{lag}'] - df[f'PL_LT_{lag}']
        new_cols[f'net_LT_SW_{lag}'] = df[f'LT_SW_{lag}'] - df[f'SW_LT_{lag}']

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    new_cols = {}
    for window in [6, 12, 24, 48, 168]:
        new_cols[f'net_LT_LV_rolling_mean_{window}'] = df[f'LT_LV_rolling_mean_{window}'] - df[f'LV_LT_rolling_mean_{window}']
        new_cols[f'net_LT_PL_rolling_mean_{window}'] = df[f'LT_PL_rolling_mean_{window}'] - df[f'PL_LT_rolling_mean_{window}']
        new_cols[f'net_LT_SW_rolling_mean_{window}'] = df[f'LT_SW_rolling_mean_{window}'] - df[f'SW_LT_rolling_mean_{window}']
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    cols_to_drop = [c for c in df.columns if c.startswith(('LT_LV','LV_LT','LT_PL','PL_LT','LT_SW','SW_LT'))]
    df = df.drop(columns=cols_to_drop)

    df.to_csv(os.path.join(CURRENT_DIR, "flows_combined.csv"), index=False)

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