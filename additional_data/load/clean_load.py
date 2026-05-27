import pandas as pd
import numpy as np
import os

CURRENT_DIR = current_dir = os.path.dirname(os.path.abspath(__file__))

def clean_data(csv_name, output_name):
    df = pd.read_csv(os.path.join(CURRENT_DIR, csv_name), sep="\t")
    df = df[df["CountryCode"] == "LT"]
    df = df.drop(["CreateDate","UpdateDate","MeasureItem", "CountryCode"], axis = 1)
    df = df.drop(["Cov_ratio"], axis = 1)
    
    df["Date"] = pd.to_datetime(df["DateShort"] + " " + df["TimeFrom"], dayfirst=True)
    df = df.sort_values("Date")
    df = df.drop(["DateShort", "TimeFrom", "TimeTo", "DateUTC", "Value_ScaleTo100"], axis=1)
    df = df.reset_index(drop=True)
    
    df["hour"] = df["Date"].dt.hour
    df["day"] = df["Date"].dt.date

    df = df.drop(["hour","day"],axis=1)
    df = df.rename(columns={"Value" : "value"})
    df = df.rename(columns={"Date" : "date"})
    df.to_csv(os.path.join(CURRENT_DIR, output_name), index=False)
    
    return df

def clean_load():
    df_2023 = clean_data("monthly_hourly_load_values_2023.csv","2023_cleaned.csv")
    df_2024 = clean_data("monthly_hourly_load_values_2024.csv","2024_cleaned.csv")
    df_2025 = clean_data("monthly_hourly_load_values_2025.csv","2025_cleaned.csv")
    
    df = pd.concat([df_2023,df_2024,df_2025])
    
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    df_filtered = df[
        df["date"].between(
            pd.Timestamp("2023-01-15 00:00:00"),
            pd.Timestamp("2025-10-05 19:00:00")
        )
    ]
    df_filtered = df_filtered.drop(["value"],axis=1)
    df_filtered = df_filtered.rename(columns={'date': 'utc'})
    path = os.path.join(CURRENT_DIR,"full_cleaned.csv")
    df_filtered.to_csv(path, index=False)
    
    return df_filtered


def create_lag_features(df, lag_hours=[1, 2, 3, 6, 12, 24, 48, 96, 168]):
    for lag in lag_hours:
        df[f'load_lag_{lag}'] = df['value'].shift(lag)

    df = remove_nan_rows(df)
    return df

def create_rolling_features(df, windows=[6, 12, 24, 48, 168]):
    shifted = df['value'].shift(1)
    for window in windows:
        df[f'load_rolling_mean_{window}'] = shifted.rolling(window=window).mean()

    df = remove_nan_rows(df)
    return df

def remove_nan_rows(df):    
    df_ml = df.dropna()
    return df_ml