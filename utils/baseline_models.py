from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

<<<<<<< HEAD
def calculate_baseline_models(X,Y):
    print("\nCreating baseline models")
    
    # Baseline 1: Last known value
    baseline1_mae = mean_absolute_error(Y, X['lag_1'])
    baseline1_rmse = np.sqrt(mean_squared_error(Y, X['lag_1']))
    
    # Baseline 2: Mean 
    y_pred_mean = np.full_like(Y, Y.mean())
    baseline2_mae = mean_absolute_error(Y, y_pred_mean)
    baseline2_rmse = np.sqrt(mean_squared_error(Y, y_pred_mean))
    
    # Baseline 3: Rolling mean of last 24 hours
    if 'rolling_mean_24' in X.columns:
        baseline3_mae = mean_absolute_error(Y, X['rolling_mean_24'])
        baseline3_rmse = np.sqrt(mean_squared_error(Y, X['rolling_mean_24']))
    else:
        baseline3_mae = baseline3_rmse = None
        print("Baseline 3: rolling_mean_24 not available")
    
    return {
        'last_value': (baseline1_mae, baseline1_rmse),
        'mean': (baseline2_mae, baseline2_rmse),
        'rolling_mean': (baseline3_mae, baseline3_rmse)
    }
    
=======
def calculate_baseline_models(X, y):
    print("\nCreating baseline models")

    X_cv, X_test, y_cv, y_test = split(X, y, test_size=0.2)

    cv_results = run_cv(X_cv, y_cv, 5, lag1_index=8, roll24_index=21)
    test_results = baseline_final(X_cv, y_cv, X_test, y_test, 8, 21)

    return {
        "cv": cv_results,
        "test": test_results
    }
def split(X, y, test_size):
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

def run_cv(X, y, n_splits, lag1_index, roll24_index):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        actual = y_val

        lag1_pred = X_val[:, lag1_index]
        mask1 = ~np.isnan(lag1_pred)
        lag1_rmse = np.sqrt(mean_squared_error(actual[mask1], lag1_pred[mask1]))
        lag1_mae  = mean_absolute_error(actual[mask1], lag1_pred[mask1])

        mean_pred = np.full(len(y_val), y_train.mean())
        mean_rmse = np.sqrt(mean_squared_error(actual, mean_pred))
        mean_mae  = mean_absolute_error(actual, mean_pred)

        roll24_pred = X_val[:, roll24_index]
        mask24 = ~np.isnan(roll24_pred)
        roll24_rmse = np.sqrt(mean_squared_error(actual[mask24], roll24_pred[mask24]))
        roll24_mae  = mean_absolute_error(actual[mask24], roll24_pred[mask24])

        print(f"\nFold {fold+1}")
        print(f"lag_1     | RMSE: {lag1_rmse:.4f} | MAE: {lag1_mae:.4f}")
        print(f"mean      | RMSE: {mean_rmse:.4f} | MAE: {mean_mae:.4f}")
        print(f"rolling_mean_24    | RMSE: {roll24_rmse:.4f} | MAE: {roll24_mae:.4f}")

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
    mask1 = ~np.isnan(lag1_pred)
    lag1_rmse = np.sqrt(mean_squared_error(actual[mask1], lag1_pred[mask1]))
    lag1_mae  = mean_absolute_error(actual[mask1], lag1_pred[mask1])

    mean_pred = np.full(len(y_test), y_train.mean())
    mean_rmse = np.sqrt(mean_squared_error(actual, mean_pred))
    mean_mae  = mean_absolute_error(actual, mean_pred)

    roll24_pred = X_test[:, roll24_index]
    mask24 = ~np.isnan(roll24_pred)
    roll24_rmse = np.sqrt(mean_squared_error(actual[mask24], roll24_pred[mask24]))
    roll24_mae  = mean_absolute_error(actual[mask24], roll24_pred[mask24])

    print("\nTest results:")
    print(f"lag_1     | RMSE: {lag1_rmse:.4f} | MAE: {lag1_mae:.4f}")
    print(f"mean      | RMSE: {mean_rmse:.4f} | MAE: {mean_mae:.4f}")
    print(f"rolling_mean_24 | RMSE: {roll24_rmse:.4f} | MAE: {roll24_mae:.4f}")

    return {
        "lag1":  (lag1_rmse, lag1_mae),
        "mean":  (mean_rmse, mean_mae),
        "rolling_mean_24": (roll24_rmse, roll24_mae),
    }
>>>>>>> 6310fa8 (changes from laptop)
