from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import pandas as pd
import optuna
import optuna.visualization.matplotlib as optuna_plot
import matplotlib.pyplot as plt

class BaseTimeSeriesModel:
    def __init__(self, feature_names=None, output_dir=None, n_splits=None, test_size=None, baseline=None):
        self.n_splits  = n_splits
        self.test_size = test_size
        self.feature_names = feature_names
        self.output_dir    = output_dir
        self.results_dir = os.path.join(self.output_dir, "results")
        self.optuna_dir = os.path.join(self.output_dir, "optuna_results")
        self.histories = []
        self.baseline = baseline

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.optuna_dir, exist_ok=True)

    def split(self, X, y):
        split_idx = int(len(X) * (1 - self.test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def build_model(self, **kwargs):
        raise NotImplementedError

    def fit_fold(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def run(self, X, y):
        X_cv, X_test, y_cv, y_test = self.split(X, y)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
            X_train, X_val = X_cv[train_idx], X_cv[val_idx]
            y_train, y_val = y_cv[train_idx], y_cv[val_idx]

            pred, actual = self.fit_fold(X_train, y_train, X_val, y_val)
                
            fold_rmse = np.sqrt(mean_squared_error(actual, pred))
            fold_mae = mean_absolute_error(actual, pred)
            fold_rmae = fold_mae / self.baseline

            print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
            print(f"Fold {fold+1} MAE: {fold_mae:.4f}")
            print(f"Fold {fold+1} rMAE: {fold_rmae:.4f}")

            results.append({
                "fold": fold + 1,
                "type": "cv",
                "rmse": fold_rmse,
                "mae": fold_mae,
                "rmae": fold_rmae
            })

        pred, actual = self.fit_final(X_cv, y_cv, X_test, y_test)
        pred = np.array(pred).flatten()
        actual = np.array(actual).flatten()
        
        test_rmse = np.sqrt(mean_squared_error(actual, pred))
        test_mae = mean_absolute_error(actual, pred)
        test_rmae = test_mae / self.baseline

        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test rMAE: {test_rmae:.4f}")

        results.append({
            "fold": "test",
            "type": "test",
            "rmse": test_rmse,
            "mae": test_mae,
            "rmae": test_rmae
        })

        cv_results    = [r for r in results if r["type"] == "cv"]
        mean_cv_rmse  = np.mean([r["rmse"] for r in cv_results])
        mean_cv_mae   = np.mean([r["mae"]  for r in cv_results])
        mean_cv_rmae  = np.mean([r["rmae"] for r in cv_results])
        
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, "metrics.csv"), index=False)

        pd.DataFrame({
            "actual":    np.array(actual).flatten(),
            "predicted": np.array(pred).flatten(),
            "error":     actual - pred
        }).to_csv(os.path.join(self.results_dir, "predictions.csv"), index=False)

        return {
            "test_rmse":    test_rmse,
            "test_mae":     test_mae,
            "test_rmae":    test_rmae,
            "mean_cv_rmse": mean_cv_rmse,
            "mean_cv_mae":  mean_cv_mae,
            "mean_cv_rmae": mean_cv_rmae,
            "predictions":  pred,
            "actuals":      actual,
            "results":      results
        }
    
    def fit_final(self, X_cv, y_cv, X_test, y_test):
        raise NotImplementedError
    
    def run_optuna(self, X, y, n_trials=None):
        X_cv, X_test, y_cv, y_test = self.split(X, y)

        def objective(trial):
            self.suggest_hyperparams(trial)
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            fold_rmses = []
            for train_idx, val_idx in tscv.split(X_cv):
                pred, actual = self.fit_fold(X_cv[train_idx], y_cv[train_idx],
                                                    X_cv[val_idx],   y_cv[val_idx])
                fold_rmses.append(np.sqrt(mean_squared_error(actual, pred)))
            return np.mean(fold_rmses)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        print("Best params:", study.best_params)
        print(f"Best CV RMSE: {study.best_value:.4f}")

        self.apply_best_params(study.best_params)

        pred, actual = self.fit_final(X_cv, y_cv, X_test, y_test)
        pred   = np.array(pred).flatten()
        actual = np.array(actual).flatten()
        test_rmse = np.sqrt(mean_squared_error(actual, pred))
        print(f"Test RMSE: {test_rmse:.4f}")

        pd.DataFrame({
            "actual":    actual.flatten(),
            "predicted": pred.flatten(),
            "error":     actual - pred
        }).to_csv(os.path.join(self.optuna_dir, "predictions.csv"), index=False)
        study.trials_dataframe().to_csv(os.path.join(self.optuna_dir, "optuna_trials.csv"), index=False)
        pd.DataFrame([{**study.best_params, "cv_rmse": study.best_value,
                    "test_rmse": test_rmse}]).to_csv(os.path.join(self.optuna_dir, "best_params.csv"), index=False)

        plt.figure(figsize=(8,6))
        fig = optuna_plot.plot_optimization_history(study)
        fig.figure.axes[0].set_xlabel("Bandymo numeris") 
        fig.figure.axes[0].set_ylabel("Paklaida (rmse)")        
        fig.figure.axes[0].set_title("")
        fig.figure.savefig(os.path.join(self.optuna_dir, "optuna_trials.png"))
        plt.close(fig.figure)

        return {
            "best_params": study.best_params,
            "cv_rmse":     study.best_value,
            "test_rmse":   test_rmse,
            "predictions": pred,
            "actuals":     actual,
            "study":       study
        }

    def suggest_hyperparams(self, trial):
        raise NotImplementedError

    def apply_best_params(self, params):
        raise NotImplementedError