import pandas as pd
from sklearn.model_selection import train_test_split

def create_time_features(df):
    df['hour'] = df['utc'].dt.hour
    df['day'] = df['utc'].dt.day
    df['month'] = df['utc'].dt.month
    df['year'] = df['utc'].dt.year
    df['dayofweek'] = df['utc'].dt.dayofweek
    df['quarter'] = df['utc'].dt.quarter
    df['dayofyear'] = df['utc'].dt.dayofyear
    df['weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    print("\nDATA WITH TIME FEATURES (first 10 rows):")
    print(df.head(10))
    
    return df

def create_lag_features(df, lag_hours=[1, 2, 3, 6, 12, 24]):
    print("\nCREATING LAG FEATURES:")
    for lag in lag_hours:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    df = remove_nan_rows(df)
    print("\nDATA WITH LAG FEATURES AFTER REMOVING NaN (first 10 rows):")
    print(df.head(10))
    return df

def create_rolling_features(df, windows=[6, 12, 24]):
    print("CREATING ROLLING FEATURES:")
    for window in windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()

    df = remove_nan_rows(df)
    print("\nDATA WITH ROLLING FEATURES AND REMOVING NaN (first 10 rows):")
    print(df.head(10))
    return df

def remove_nan_rows(df):    
    rows_before = df.shape[0]
    print(f"Rows before dropping NaN: {rows_before}")
    print("\nNaN values per column:")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0])
    
    df_ml = df.dropna()
    
    rows_after = df_ml.shape[0]
    rows_dropped = rows_before - rows_after
    percent_dropped = (rows_dropped / rows_before) * 100
    
    print(f"\nRows after dropping NaN: {rows_after}")
    print(f"Rows dropped: {rows_dropped}")
    print(f"Percentage dropped: {percent_dropped:.2f}%")
    print("\n" + "=" * 60)
    
    return df_ml

def prepare_train_test_data(df_ml):
    X = df_ml.drop(['value', 'utc'], axis=1)
    y = df_ml['value']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X, y, X_train, X_test, y_train, y_test

