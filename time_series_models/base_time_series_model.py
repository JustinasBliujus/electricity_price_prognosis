from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

class BaseTimeSeriesModel:
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits  = n_splits
        self.test_size = test_size

    def split(self, X, y):
        split_idx = int(len(X) * (1 - self.test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def build_model(self, **kwargs):
        raise NotImplementedError

    def fit_fold(self, model, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def run(self, X, y):
        X_cv, X_test, y_cv, y_test = self.split(X, y)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_rmses = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
            X_train, X_val = X_cv[train_idx], X_cv[val_idx]
            y_train, y_val = y_cv[train_idx], y_cv[val_idx]

            n_features = X_train.shape[1]
            model = self.build_model(n_features)
            pred, actual = self.fit_fold(model, X_train, y_train, X_val, y_val)

            fold_rmse = np.sqrt(mean_squared_error(actual, pred))
            fold_rmses.append(fold_rmse)
            print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

        n_features = X_train.shape[1]
        model = self.build_model(n_features)
        pred_test, actual_test = self.fit_final(model, X_cv, y_cv, X_test, y_test)
        test_rmse = np.sqrt(mean_squared_error(actual_test, pred_test))
        print(f"Test RMSE: {test_rmse:.4f}")

        return {
            "fold_rmses":    fold_rmses,
            "mean_cv_rmse":  np.mean(fold_rmses),
            "test_rmse":     test_rmse,
            "predictions":   pred_test,
            "actuals":       actual_test,
            "model":         model
        }

    def fit_final(self, model, X_cv, y_cv, X_test, y_test):
        raise NotImplementedError