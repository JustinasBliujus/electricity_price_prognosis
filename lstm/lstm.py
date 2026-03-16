import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from .build_lstm import build_lstm
from time_series_models.neural_time_series_model import NeuralTimeSeriesModel
import os

EarlyStopping = tf.keras.callbacks.EarlyStopping

DIR = os.path.dirname(os.path.abspath(__file__))

class LSTMModel(NeuralTimeSeriesModel):
    def __init__(self, n_lstm_layers=2, units_per_layer=None, dropout_per_layer=None,
                 dense_units=32, learning_rate=0.001, **kwargs):
        super().__init__(**kwargs)
        self.n_lstm_layers     = n_lstm_layers
        self.units_per_layer   = units_per_layer or [128, 64]
        self.dropout_per_layer = dropout_per_layer or [0.2, 0.2]
        self.dense_units       = dense_units
        self.learning_rate     = learning_rate

    def build_model(self, n_features):
        return build_lstm(n_features=n_features, n_lstm_layers=self.n_lstm_layers,
                          units_per_layer=self.units_per_layer,
                          dropout_per_layer=self.dropout_per_layer,
                          dense_units=self.dense_units,
                          learning_rate=self.learning_rate)

    def fit_fold(self, model, X_train, y_train, X_val, y_val):
        X_scaler = self.X_scaler_class()
        y_scaler = self.y_scaler_class()

        X_train_s = X_scaler.fit_transform(X_train)
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_val_s   = X_scaler.transform(X_val)
        y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        n_features = X_train_s.shape[1]
        X_train_s  = X_train_s.reshape((X_train_s.shape[0], 1, n_features))
        X_val_s    = X_val_s.reshape((X_val_s.shape[0], 1, n_features))

        self.y_scaler = y_scaler

        model = self.build_model(n_features)
        model.fit(X_train_s, y_train_s, epochs=self.epochs, batch_size=self.batch_size,
                  validation_data=(X_val_s, y_val_s),
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=0)

        pred = y_scaler.inverse_transform(model.predict(X_val_s))
        return pred, y_val

    def fit_final(self, model, X_cv, y_cv, X_test, y_test):
        X_scaler = self.X_scaler_class()
        y_scaler = self.y_scaler_class()

        X_cv_s   = X_scaler.fit_transform(X_cv)
        y_cv_s   = y_scaler.fit_transform(y_cv.reshape(-1, 1)).flatten()
        X_test_s = X_scaler.transform(X_test)

        n_features = X_cv_s.shape[1]
        X_cv_s   = X_cv_s.reshape((X_cv_s.shape[0], 1, n_features))
        X_test_s = X_test_s.reshape((X_test_s.shape[0], 1, n_features))

        model = self.build_model(n_features)
        model.fit(X_cv_s, y_cv_s, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        pred = y_scaler.inverse_transform(model.predict(X_test_s))
        return pred, y_test

    def run_optuna(self, X, y, n_trials=100):
        X_cv, X_test, y_cv, y_test = self.split(X, y)

        def objective(trial):
            self.n_lstm_layers     = trial.suggest_int("n_lstm_layers", 1, 3)
            self.batch_size        = trial.suggest_categorical("batch_size", [16, 32, 64])
            self.learning_rate     = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            self.dense_units       = trial.suggest_int("dense_units", 16, 128, step=16)
            self.units_per_layer   = [trial.suggest_int(f"units_{i}", 32, 256, step=32) for i in range(self.n_lstm_layers)]
            self.dropout_per_layer = [trial.suggest_float(f"dropout_{i}", 0.1, 0.5) for i in range(self.n_lstm_layers)]

            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            fold_rmses = []

            for train_idx, val_idx in tscv.split(X_cv):
                X_train, X_val = X_cv[train_idx], X_cv[val_idx]
                y_train, y_val = y_cv[train_idx], y_cv[val_idx]

                pred, actual = self.fit_fold(None, X_train, y_train, X_val, y_val)
                fold_rmses.append(np.sqrt(mean_squared_error(actual, pred)))

            return np.mean(fold_rmses)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        print("Best params:", study.best_params)
        print(f"Best CV RMSE: {study.best_value:.4f}")

        p = study.best_params
        self.n_lstm_layers     = p["n_lstm_layers"]
        self.learning_rate     = p["learning_rate"]
        self.dense_units       = p["dense_units"]
        self.batch_size        = p["batch_size"]
        self.units_per_layer   = [p[f"units_{i}"] for i in range(p["n_lstm_layers"])]
        self.dropout_per_layer = [p[f"dropout_{i}"] for i in range(p["n_lstm_layers"])]

        testPredict, testY = self.fit_final(None, X_cv, y_cv, X_test, y_test)
        testPredict = np.array(testPredict).flatten()
        testY       = np.array(testY).flatten()
        test_rmse   = np.sqrt(mean_squared_error(testY, testPredict))
        print(f"Test RMSE: {test_rmse:.4f}")

        pd.DataFrame({
            "actual":    testY,
            "predicted": testPredict,
            "error":     testY - testPredict
        }).to_csv(os.path.join(DIR, "predictions.csv"), index=False)
        study.trials_dataframe().to_csv(os.path.join(DIR, "optuna_trials.csv"), index=False)
        pd.DataFrame([{**p, "cv_rmse": study.best_value,
                    "test_rmse": test_rmse}]).to_csv(os.path.join(DIR, "best_params.csv"), index=False)

        return {
            "best_params": p,
            "cv_rmse":     study.best_value,
            "test_rmse":   test_rmse,
            "predictions": testPredict,
            "actuals":     testY,
            "study":       study
        }

def lstm(X, y, n_splits=5, test_size=0.2, epochs=100, batch_size=32):
    return LSTMModel(epochs=epochs, batch_size=batch_size,
                     n_splits=n_splits, test_size=test_size).run(X, y)

def lstm_optuna(X, y, n_splits=5, test_size=0.2, n_trials=100, epochs=100):
    return LSTMModel(epochs=epochs, n_splits=n_splits,
                     test_size=test_size).run_optuna(X, y, n_trials=n_trials)