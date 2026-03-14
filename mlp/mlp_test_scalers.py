from sklearn.preprocessing import MinMaxScaler, RobustScaler
from mlp.mlp import mlp
import os
import pandas as pd
from datetime import datetime

CATEGORICAL_COLS = ['hour', 'day', 'month', 'year', 'dayofweek', 'quarter', 'dayofyear', 'weekend']
CONTINUOUS_COLS = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
                   'rolling_mean_6', 'rolling_std_6', 'rolling_mean_12',
                   'rolling_std_12', 'rolling_mean_24', 'rolling_std_24']

def mlp_test_scalers(X, y, scale_all=True):
    results = {}

    for scaler_name, scaler_X_class, scaler_y_class in [
        ("minmax", MinMaxScaler, MinMaxScaler),
        ("robust", RobustScaler, RobustScaler),
    ]:
        print(f"Running with {scaler_name} scaler")

        results[scaler_name] = mlp(X, y, n_splits=5, test_size=0.2, epochs=100, batch_size=32,
        X_scaler_class = scaler_X_class, y_scaler_class = scaler_y_class, scale_all=scale_all)

        save_results(scaler_name, results[scaler_name], scale_all)
        
        if scale_all:
            print("\nScaled all features")
        else:
            print("\nScaled only continuous features")
    for scaler_name, result in results.items():
        print(f"\nScaler: {scaler_name} (Test_rmse: {result['test_rmse']:.4f}) (Mean_cv_rmse: {result['mean_cv_rmse']:.4f})")
        
    return results


CSV_PATH = "mlp_test_scalers_results.csv"

def save_results(scaler_name, result, scale_all):
    row = {
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scaler":        scaler_name,
        "scale_all":     scale_all,
        "test_rmse":     result["test_rmse"],
        "mean_cv_rmse":  result["mean_cv_rmse"],
    }

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(CSV_PATH, index=False)