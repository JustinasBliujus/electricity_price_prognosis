from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calculate_baseline_models(y_train, y_test, X_test):
    print("\nCreating baseline models")
    
    # Baseline 1: Last known value
    y_pred_last = y_test.shift(1).fillna(y_train.iloc[-1])
    baseline1_mae = mean_absolute_error(y_test, y_pred_last)
    baseline1_rmse = np.sqrt(mean_squared_error(y_test, y_pred_last))
    baseline1_r2 = r2_score(y_test, y_pred_last)
    
    # Baseline 2: Mean of training data
    y_pred_mean = np.full_like(y_test, y_train.mean())
    baseline2_mae = mean_absolute_error(y_test, y_pred_mean)
    baseline2_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    baseline2_r2 = r2_score(y_test, y_pred_mean)
    
    # Baseline 3: Rolling mean of last 24 hours
    if 'rolling_mean_24' in X_test.columns:
        y_pred_rolling = X_test['rolling_mean_24']
        baseline3_mae = mean_absolute_error(y_test, y_pred_rolling)
        baseline3_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rolling))
        baseline3_r2 = r2_score(y_test, y_pred_rolling)
    else:
        baseline3_mae = baseline3_rmse = baseline3_r2 = None
        print("Baseline 3: rolling_mean_24 not available")
    
    return {
        'last_value': (baseline1_mae, baseline1_rmse, baseline1_r2),
        'train_mean': (baseline2_mae, baseline2_rmse, baseline2_r2),
        'rolling_mean': (baseline3_mae, baseline3_rmse, baseline3_r2)
    }