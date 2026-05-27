import os
from utils.data_preprocessing import load_raw_data, clean_data, handle_datetime_issues
from utils.feature_engineering import (create_time_features, create_lag_features, 
                               create_rolling_features, prepare_train_test_data)

def prepare_data():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    folder_path_datasheets = 'datasheets'
    os.makedirs(folder_path_datasheets, exist_ok=True)
    folder_path_visualizations = 'visualizations'
    os.makedirs(folder_path_visualizations, exist_ok=True)
    
    df_prices = load_raw_data('elektros-energijos-kainos-801.csv')
    print(df_prices.shape)
    df_clean = clean_data(df_prices)
    print(df_clean.shape)
    df_clean = handle_datetime_issues(df_clean)
    print(df_clean.shape)
    df_clean = create_time_features(df_clean)
    print(df_clean.shape)
    df_clean = create_lag_features(df_clean)
    print(df_clean.shape)
    df_clean = create_rolling_features(df_clean)
    print(df_clean.shape)
    X, y, X_train, X_test, y_train, y_test = prepare_train_test_data(df_clean)
    
    # print(df_clean['value'].std())
    # print(df_clean['value'].min())
    # print(df_clean['value'].max())
    # print(df_clean.groupby('weekend')['value'].mean())
    # print(df_clean[df_clean['hour'].isin([11, 17])].groupby('hour')['value'].agg(['mean', 'std']))

    df_clean.to_csv(os.path.join(folder_path_datasheets, 'electricity_final.csv'), index=False)
    X.to_csv(os.path.join(folder_path_datasheets, 'electricity_prices_features.csv'), index=False)
    y.to_csv(os.path.join(folder_path_datasheets, 'electricity_prices_target.csv'), index=False)
    X_train.to_csv(os.path.join(folder_path_datasheets, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(folder_path_datasheets, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(folder_path_datasheets, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(folder_path_datasheets, 'y_test.csv'), index=False)

    print("data preparation done")

    return df_clean, X, y, X_train, X_test, y_train, y_test
