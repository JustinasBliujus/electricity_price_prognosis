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
    
    print("\ndata with time features:")
    print(df.head(10))
    
    return df

def create_lag_features(df, lag_hours=[1, 2, 3, 6, 12, 24, 48, 96, 168]):
    for lag in lag_hours:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    df = remove_nan_rows(df)
    print("\ndata with lag features, removed nans:")
    print(df.head(10))
    return df

def create_rolling_features(df, windows=[6, 12, 24, 48, 168]):
    shifted = df['value'].shift(1)
    for window in windows:
        df[f'rolling_mean_{window}'] = shifted.rolling(window=window).mean()
        df[f'rolling_std_{window}'] = shifted.rolling(window=window).std()

    df = remove_nan_rows(df)
    print("\ndata with rolling features, removed nans:")
    print(df.head(10))
    return df

def remove_nan_rows(df):    
    rows_before = df.shape[0]
    print(f"rows before dropping nans: {rows_before}")
    print("\nnans per column:")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0])
    
    df_ml = df.dropna()
    
    rows_after = df_ml.shape[0]
    
    print(f"\nafter dropping nans: {rows_after}")
    
    return df_ml

def prepare_train_test_data(df_ml):
    X = df_ml.drop(['value', 'utc'], axis=1)
    y = df_ml['value']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"train size: {len(X_train)} ({len(X_train)/len(X)*100}%)")
    print(f"test size: {len(X_test)} ({len(X_test)/len(X)*100}%)")
    
    return X, y, X_train, X_test, y_train, y_test