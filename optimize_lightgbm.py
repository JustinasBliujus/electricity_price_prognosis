import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna

def optimize_lightgbm(df, n_trials=100, metric='mae', n_splits=5, test_size=0.1, loss_function=None, alpha_min = None, alpha_max = None):  
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    splits = time_series_split_window(df, n_splits=n_splits, test_size=test_size)
    print(f"\nCreated {len(splits)} time series splits\n")
    print(f"Used loss function: {loss_function}\n")
    print(f"Alpha range: [{alpha_min}, {alpha_max}]\n")
    print(f"Optimizing for metric: {metric.upper()}\n")

    def objective(trial):
        val_errors = []

        for fold, split in enumerate(splits):
            dtrain = lgb.Dataset(split['X_train'], label=split['y_train'])
            dval = lgb.Dataset(split['X_val'], label=split['y_val'], reference=dtrain)
        
            param = {
                "objective": loss_function,
                "metric": "mae" if metric == 'mae' else 'rmse',
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", -1, 50),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "verbose": -1,
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }
            
            if loss_function == "huber":
                param["alpha"] = trial.suggest_float("huber_alpha", alpha_min, alpha_max) 
            elif loss_function == "fair":
                param["c"] = trial.suggest_float("fair_c", alpha_min, alpha_max)
            
            gbm = lgb.train(
                param, 
                dtrain,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        
            preds = gbm.predict(dval, num_iteration=gbm.best_iteration)

            if metric == 'mae':
                fold_error = mean_absolute_error(split['y_val'], preds)
            else:
                fold_error = np.sqrt(mean_squared_error(split['y_val'], preds))
            
            val_errors.append(fold_error)

        return np.mean(val_errors)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    trial = study.best_trial

    return study, trial.params, trial.value

def time_series_split_window(df, n_splits=5, test_size=0.1):
    df = df.sort_values('utc').reset_index(drop=True)
    X = df.drop(['value','utc'], axis=1)
    y = df['value']
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(df)*test_size))
    splits = []
    for train_index, val_index in tscv.split(X):
        splits.append({
            'X_train': X.iloc[train_index],
            'y_train': y.iloc[train_index],
            'X_val': X.iloc[val_index],
            'y_val': y.iloc[val_index]
        })
        print(f"Train indices: {train_index[0]} to {train_index[-1]}, Validation indices: {val_index[0]} to {val_index[-1]}")

    return splits