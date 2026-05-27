import os
import pandas as pd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_gas(input_name, output_name, column_new_name="gas"):
    df = pd.read_csv(
        os.path.join(CURRENT_DIR, input_name),
        sep=','
    )
    
    df = df[['Date', 'Price']]
    df = df.rename(columns={'Date': 'utc', 'Price': column_new_name})

    df['utc'] = pd.to_datetime(df['utc'], format='%m/%d/%Y')
    
    df = df.sort_values('utc').reset_index(drop=True)
    df = handle_datetime_issues(df)
   
    df = df.set_index('utc')
    df = df.resample('h').ffill()
    df = df.reset_index()
    df = create_lag_features(df, column_new_name)
    df = df.drop([column_new_name], axis=1)
    df = df[df['utc'].between('2023-01-15 00:00:00', '2025-10-05 19:00:00')]
    df.to_csv(os.path.join(CURRENT_DIR, output_name), index=False)
    return df

def create_lag_features(df, col):
    for days in [1, 2, 7, 14, 30]:
        df[f'{col}_lag_{days}d'] = df[col].shift(24 * days)
    
    for days in [7, 14, 30]:
        df[f'{col}_rolling_mean_{days}d'] = df[col].shift(24).rolling(24 * days).mean()

    df = df.dropna()
    return df

def handle_datetime_issues(df):
    df = df.sort_values('utc').reset_index(drop=True)
    df = df.set_index('utc')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_range)

    df = df.interpolate(method='linear')
    df = df.reset_index().rename(columns={'index': 'utc'})

    return df