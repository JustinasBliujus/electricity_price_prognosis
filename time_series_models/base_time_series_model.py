from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import pandas as pd
import optuna
import optuna.visualization.matplotlib as optuna_plot
import matplotlib.pyplot as plt
from plot_style import PlotStyle

class BaseTimeSeriesModel:
    def __init__(self, feature_names=None, output_dir=None, n_splits=None, test_size=None):
        self.n_splits  = n_splits
        self.test_size = test_size
        self.feature_names = feature_names
        self.output_dir    = output_dir
        self.results_dir = os.path.join(self.output_dir, "results")
        self.optuna_dir = os.path.join(self.output_dir, "optuna_results")
        self.plot_style = PlotStyle()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.optuna_dir, exist_ok=True)

    def split(self, X, y):
        split_index = int(len(X) * (1 - self.test_size))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def build_model(self, **kwargs):
        raise NotImplementedError

    def fit_fold(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def run(self, X, y):
        X_cv, X_test, y_cv, y_test = self.split(X, y)
        
        time_series_split_indexes = TimeSeriesSplit(n_splits=self.n_splits)
        results = []
        all_residuals = []
        for fold, (train_index, val_index) in enumerate(time_series_split_indexes.split(X_cv)):
            X_train, X_val = X_cv[train_index], X_cv[val_index]
            y_train, y_val = y_cv[train_index], y_cv[val_index]

            pred, actual = self.fit_fold(X_train, y_train, X_val, y_val)
            pred   = np.array(pred).flatten()   
            actual = np.array(actual).flatten()
            
            fold_rmse = np.sqrt(mean_squared_error(actual, pred))
            fold_mae = mean_absolute_error(actual, pred)

            print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
            print(f"Fold {fold+1} MAE: {fold_mae:.4f}")

            results.append({
                "fold": fold + 1,
                "type": "cv",
                "rmse": fold_rmse,
                "mae": fold_mae,
            })
            residual = actual - pred

            all_residuals.append(pd.DataFrame({
                "fold": fold + 1,
                "actual": actual,
                "pred": pred,
                "residual": residual
            }))

        pred, actual = self.fit_final(X_cv, y_cv, X_test, y_test)
        pred = np.array(pred).flatten()
        actual = np.array(actual).flatten()
        test_rmse = np.sqrt(mean_squared_error(actual, pred))
        test_mae = mean_absolute_error(actual, pred)

        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")

        results.append({
            "fold": "test",
            "type": "test",
            "rmse": test_rmse,
            "mae": test_mae,
        })

        cv_results    = [r for r in results if r["type"] == "cv"]
        mean_cv_rmse  = np.mean([r["rmse"] for r in cv_results])
        mean_cv_mae   = np.mean([r["mae"]  for r in cv_results])
        
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, "metrics.csv"), index=False)

        pd.DataFrame({
            "actual":    np.array(actual).flatten(),
            "predicted": np.array(pred).flatten(),
            "error":     actual - pred
        }).to_csv(os.path.join(self.results_dir, "predictions.csv"), index=False)
        
        cv_residuals_df = pd.concat(all_residuals)
        cv_residuals_df.to_csv(os.path.join(self.results_dir, "cv_residuals.csv"), index=False)
        
        return {
            "test_rmse":    test_rmse,
            "test_mae":     test_mae,
            "mean_cv_rmse": mean_cv_rmse,
            "mean_cv_mae":  mean_cv_mae,
            "predictions":  pred,
            "actuals":      actual,
            "model":        self 
        }
    
    def fit_final(self, X_cv, y_cv, X_test, y_test):
        raise NotImplementedError
    
    def run_optuna(self, X, y, n_trials=None):
        X_cv, X_test, y_cv, y_test = self.split(X, y)
        early_stop = EarlyStoppingCallback(patience=100)
        time_series_split_indexes = TimeSeriesSplit(n_splits=self.n_splits)
        fold_indexes = list(time_series_split_indexes.split(X_cv))
        
        def objective(trial):
            self.suggest_hyperparams(trial)
            fold_rmses = []
            for train_index, val_index in fold_indexes:
                pred, actual = self.fit_fold(X_cv[train_index], y_cv[train_index], X_cv[val_index], y_cv[val_index])
                fold_rmses.append(np.sqrt(mean_squared_error(actual, pred)))
            return np.mean(fold_rmses)

        live_best_path = os.path.join(self.optuna_dir, "best_params_live.csv")
        save_callback = SaveBestParamsCallback(live_best_path)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials,callbacks=[early_stop,save_callback])
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

    def plot_fold_predictions(self, X, y, fold_number=2):
        X_cv, X_test, y_cv, y_test = self.split(X, y)
        time_series_split_indexes = TimeSeriesSplit(n_splits=self.n_splits)
        fold_indexes = list(time_series_split_indexes.split(X_cv))

        train_index, val_index = fold_indexes[fold_number - 1]

        if hasattr(X_cv, 'iloc'):
            X_train, X_val = X_cv.iloc[train_index], X_cv.iloc[val_index]
            y_train, y_val = y_cv.iloc[train_index], y_cv.iloc[val_index]
        else:
            X_train, X_val = X_cv[train_index], X_cv[val_index]
            y_train, y_val = y_cv[train_index], y_cv[val_index]

        pred, actual = self.fit_fold(X_train, y_train, X_val, y_val)
        pred = np.array(pred).flatten()
        actual = np.array(actual).flatten()

        if hasattr(X_val, "iloc"):
            dates = pd.to_datetime({
                "year":  X_val["year"].values,
                "month": X_val["month"].values,
                "day":   X_val["day"].values,
                "hour":  X_val["hour"].values if "hour" in X_val.columns else 0
            })
        else:
            dates = np.arange(len(actual))

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(dates, actual, label="Faktinė kaina")
        ax.plot(dates, pred, label="Prognozė")
        n_ticks = 5
        indexes = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)

        ax.set_xticks(dates[indexes])
        ax.set_xticklabels([pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates[indexes]])
        ax.set_xlabel("Data", fontsize=self.plot_style.label_size)
        ax.set_ylabel("Kaina (EUR/MWh)", fontsize=self.plot_style.label_size)
        self.plot_style.apply(fig, ax)

        path = os.path.join(self.results_dir, f"fold_{fold_number}_predictions.png")
        fig.savefig(path, dpi=self.plot_style.dpi, bbox_inches="tight")

        df = pd.DataFrame({
            "date": dates,
            "actual": actual,
            "predicted": pred
        })
        csv_path = os.path.join(self.results_dir, f"fold_{fold_number}_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(path)
    
class EarlyStoppingCallback:
    def __init__(self, patience=100):
        self.patience = patience
        self.best_value = None
        self.no_improve_count = 0

    def __call__(self, study, trial):
        if self.best_value is None or study.best_value < self.best_value:
            self.best_value = study.best_value
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"stopping after {self.patience} trials")
            study.stop()
            
class SaveBestParamsCallback:
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_value = None

    def __call__(self, study, trial):
        if self.best_value is None or study.best_value < self.best_value:
            self.best_value = study.best_value

            df = pd.DataFrame([{
                **study.best_params,
                "cv_rmse": study.best_value,
                "trial": trial.number
            }])

            df.to_csv(self.save_path, index=False)