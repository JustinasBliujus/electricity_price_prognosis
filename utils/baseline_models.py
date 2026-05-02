from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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
    
