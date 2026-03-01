import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import optuna

def optimize_xgboost(df, n_trials=100, metric='mae', n_splits=5, test_size=0.1, loss_function=None):  
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    splits = time_series_split_window(df, n_splits=n_splits, test_size=test_size)
    print(f"\nCreated {len(splits)} time series splits\n")
    print(f"Used loss function: {loss_function}\n")
    print(f"Optimizing for metric: {metric.upper()}\n")
    
    def objective(trial):
        val_errors = []

        for fold, split in enumerate(splits):
            dtrain = xgb.DMatrix(split['X_train'].values, label=split['y_train'].values, feature_names=list(split['X_train'].columns))
            dval = xgb.DMatrix(split['X_val'].values, label=split['y_val'].values, feature_names=list(split['X_val'].columns))
        
            param = {
                "objective": loss_function,
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 1, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  
                "gamma": trial.suggest_float("gamma", 0.0, 5.0), 
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),  
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),  
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),  
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),  
                "eval_metric": ["rmse", "mae"],
                "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "verbosity": 0,
                "seed": 42,
            }
            
            xgboost = xgb.train(
                param, 
                dtrain,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False
            )
        
            preds = xgboost.predict(dval)

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