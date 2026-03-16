from .build_xgboost import build_xgboost
from time_series_models.base_time_series_model import BaseTimeSeriesModel
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from .build_xgboost import build_xgboost
import os

DIR = os.path.dirname(os.path.abspath(__file__))

class XGBModel(BaseTimeSeriesModel):
    def __init__(self, objective='reg:squarederror', n_estimators=500,
                 learning_rate=0.01, max_depth=6, colsample_bytree=0.8, **kwargs):
        super().__init__(**kwargs)
        self.objective       = objective
        self.n_estimators    = n_estimators
        self.learning_rate   = learning_rate
        self.max_depth       = max_depth
        self.colsample_bytree = colsample_bytree

    def build_model(self, n_features = None):
        return build_xgboost(n_estimators=self.n_estimators,
                             learning_rate=self.learning_rate,
                             max_depth=self.max_depth,
                             colsample_bytree=self.colsample_bytree,
                             objective=self.objective)

    def fit_fold(self, model, X_train, y_train, X_val, y_val):
        model.fit(X_train, y_train)
        return model.predict(X_val), y_val

    def fit_final(self, model, X_cv, y_cv, X_test, y_test):
        model.fit(X_cv, y_cv)
        return model.predict(X_test), y_test

    def run_optuna(self, X, y, n_trials=100):
        X_cv, X_test, y_cv, y_test = self.split(X, y)

        def objective(trial):
            self.n_estimators     = trial.suggest_int("n_estimators", 100, 1000, step=100)
            self.learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            self.max_depth        = trial.suggest_int("max_depth", 3, 10)
            self.colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            fold_rmses = []

            for train_idx, val_idx in tscv.split(X_cv):
                X_train, X_val = X_cv[train_idx], X_cv[val_idx]
                y_train, y_val = y_cv[train_idx], y_cv[val_idx]

                model = self.build_model()
                pred, actual = self.fit_fold(model, X_train, y_train, X_val, y_val)
                fold_rmses.append(np.sqrt(mean_squared_error(actual, pred)))

            return np.mean(fold_rmses)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("Best params:", study.best_params)
        print(f"Best CV RMSE: {study.best_value:.4f}")

        p = study.best_params
        self.n_estimators     = p["n_estimators"]
        self.learning_rate    = p["learning_rate"]
        self.max_depth        = p["max_depth"]
        self.colsample_bytree = p["colsample_bytree"]

        best_model = self.build_model()
        best_model.fit(X_cv, y_cv)

        testPredict = best_model.predict(X_test)
        test_rmse   = np.sqrt(mean_squared_error(y_test, testPredict))
        print(f"Test RMSE: {test_rmse:.4f}")

        pd.DataFrame({
            "actual":    y_test.flatten(),
            "predicted": testPredict.flatten(),
            "error":     (y_test - testPredict).flatten()
        }).to_csv(os.path.join(DIR, "predictions.csv"), index=False)
        study.trials_dataframe().to_csv(os.path.join(DIR, "optuna_trials.csv"), index=False)
        pd.DataFrame([{**p, "cv_rmse": study.best_value,
                    "test_rmse": test_rmse}]).to_csv(os.path.join(DIR, "best_params.csv"), index=False)

        return {
            "best_params": p,
            "cv_rmse":     study.best_value,
            "test_rmse":   test_rmse,
            "predictions": testPredict,
            "actuals":     y_test,
            "model":       best_model,
            "study":       study
        }

def xgb(X, y, n_splits=5, test_size=0.2, objective='reg:squarederror'):
    return XGBModel(objective=objective, n_splits=n_splits,
                    test_size=test_size).run(X, y)

def xgboost_optuna(X, y, n_splits=5, test_size=0.2, n_trials=100):
    return XGBModel(n_splits=n_splits, test_size=test_size).run_optuna(X, y, n_trials=n_trials)