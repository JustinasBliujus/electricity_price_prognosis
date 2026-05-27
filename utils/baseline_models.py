from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def calculate_baseline_models(X, y): 
    X = np.asarray(X)
    y = np.asarray(y)      
    X_cv, X_test, y_cv, y_test = split(X, y, test_size=0.2)

    cv_results = run_cv(X_cv, y_cv, 5, lag1_index=8, roll24_index=21)
    test_results = baseline_final(X_cv, y_cv, X_test, y_test, 8, 21)

    return {
        "cv": cv_results,
        "test": test_results
    }
    
def split(X, y, test_size):
        split_index = int(len(X) * (1 - test_size))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def run_cv(X, y, n_splits, lag1_index, roll24_index):
    time_series_split_indexes = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_index, val_index) in enumerate(time_series_split_indexes.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        actual = y_val

        lag1_pred = X_val[:, lag1_index]
        lag1_rmse = np.sqrt(mean_squared_error(actual, lag1_pred))
        lag1_mae  = mean_absolute_error(actual, lag1_pred)

        mean_pred = np.full(len(y_val), y_train.mean())
        mean_rmse = np.sqrt(mean_squared_error(actual, mean_pred))
        mean_mae  = mean_absolute_error(actual, mean_pred)

        roll24_pred = X_val[:, roll24_index]
        roll24_rmse = np.sqrt(mean_squared_error(actual, roll24_pred))
        roll24_mae  = mean_absolute_error(actual, roll24_pred)

        results.append({
            "fold": fold + 1,
            "lag1_rmse": lag1_rmse,
            "lag1_mae": lag1_mae,
            "mean_rmse": mean_rmse,
            "mean_mae": mean_mae,
            "rolling_mean_24_rmse": roll24_rmse,
            "rolling_mean_24_mae": roll24_mae,
        })

    return results

def baseline_final(X_train, y_train, X_test, y_test, lag1_index, roll24_index):
    actual = y_test

    lag1_pred = X_test[:, lag1_index]
    lag1_rmse = np.sqrt(mean_squared_error(actual, lag1_pred))
    lag1_mae  = mean_absolute_error(actual, lag1_pred)

    mean_pred = np.full(len(y_test), y_train.mean())
    mean_rmse = np.sqrt(mean_squared_error(actual, mean_pred))
    mean_mae  = mean_absolute_error(actual, mean_pred)

    roll24_pred = X_test[:, roll24_index]
    roll24_rmse = np.sqrt(mean_squared_error(actual, roll24_pred))
    roll24_mae  = mean_absolute_error(actual, roll24_pred)

    return {
        "lag1":  (lag1_rmse, lag1_mae),
        "mean":  (mean_rmse, mean_mae),
        "rolling_mean_24": (roll24_rmse, roll24_mae),
    }
