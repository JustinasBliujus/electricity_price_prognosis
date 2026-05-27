import os
import pandas as pd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_weather():
    df = pd.read_csv(os.path.join(CURRENT_DIR, "weather.csv"), sep=",")
    df['utc'] = pd.to_datetime(df['utc'])
    df = df.sort_values('utc').reset_index(drop=True)

    full_range = pd.date_range(start=df['utc'].min(), end=df['utc'].max(), freq='h')
    missing = full_range.difference(df['utc'])
    print(missing)

    if len(missing) > 0:
        df = df.set_index('utc')
        df = df.reindex(full_range)
        df = df.interpolate(method='linear')
        df = df.reset_index().rename(columns={'index': 'utc'})

    weather_cols = [c for c in df.columns if c != 'utc']
    
    for col in weather_cols:
        for hours in [1, 24, 48, 168]:
            df[f'{col}_lag_{hours}h'] = df[col].shift(hours)
        for window in [24, 168]:
            df[f'{col}_rolling_mean_{window}h'] = df[col].shift(1).rolling(window).mean()
        df = df.drop([col],axis=1)

    df = df.dropna()
    df = df[df['utc'].between('2023-01-15 00:00:00', '2025-10-05 19:00:00')]
    full_range = pd.date_range(start=df['utc'].min(), end=df['utc'].max(), freq='h')
    missing = full_range.difference(df['utc'])
    print(missing)
    df.to_csv(os.path.join(CURRENT_DIR, "weather_cleaned.csv"), index=False)
    return df