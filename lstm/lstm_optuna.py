import optuna
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import tensorflow as tf
import pandas as pd
from build_lstm import build_lstm
Sequential    = tf.keras.models.Sequential
Dense         = tf.keras.layers.Dense
Dropout       = tf.keras.layers.Dropout
Input         = tf.keras.layers.Input
LSTM          = tf.keras.layers.LSTM
EarlyStopping = tf.keras.callbacks.EarlyStopping

def lstm_optuna(X, y, n_splits=5, test_size=0.2, n_trials=100, epochs=100):
    split_idx = int(len(X) * (1 - test_size))
    X_cv, X_test = X[:split_idx], X[split_idx:]
    y_cv, y_test = y[:split_idx], y[split_idx:]

    def objective(trial):
        n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 3)
        batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dense_units   = trial.suggest_int("dense_units", 16, 128, step=16)

        units_per_layer   = [trial.suggest_int(f"units_{i}", 32, 256, step=32) for i in range(n_lstm_layers)]
        dropout_per_layer = [trial.suggest_float(f"dropout_{i}", 0.1, 0.5) for i in range(n_lstm_layers)]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_rmses = []

        for train_idx, val_idx in tscv.split(X_cv):
            X_train, X_val = X_cv[train_idx], X_cv[val_idx]
            y_train, y_val = y_cv[train_idx], y_cv[val_idx]

            X_scaler = RobustScaler()
            y_scaler = RobustScaler()

            X_train_s = X_scaler.fit_transform(X_train)
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            X_val_s   = X_scaler.transform(X_val)
            y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

            n_features = X_train_s.shape[1]
            X_train_s  = X_train_s.reshape((X_train_s.shape[0], 1, n_features))
            X_val_s    = X_val_s.reshape((X_val_s.shape[0], 1, n_features))

            model = build_lstm(n_features, n_lstm_layers, units_per_layer, dropout_per_layer, dense_units, learning_rate)

            model.fit(
                X_train_s, y_train_s,
                epochs=100,
                batch_size=batch_size,
                validation_data=(X_val_s, y_val_s),
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0,
            )

            pred   = y_scaler.inverse_transform(model.predict(X_val_s))
            actual = y_scaler.inverse_transform(y_val_s.reshape(-1, 1))
            fold_rmses.append(np.sqrt(mean_squared_error(actual, pred)))

        return np.mean(fold_rmses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best params:", study.best_params)
    print(f"Best CV RMSE: {study.best_value:.4f}")

    p = study.best_params
    X_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_cv_s   = X_scaler.fit_transform(X_cv)
    y_cv_s   = y_scaler.fit_transform(y_cv.reshape(-1, 1)).flatten()
    X_test_s = X_scaler.transform(X_test)
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    n_features = X_cv_s.shape[1]
    X_cv_s   = X_cv_s.reshape((X_cv_s.shape[0], 1, n_features))
    X_test_s = X_test_s.reshape((X_test_s.shape[0], 1, n_features))

    units_per_layer   = [p[f"units_{i}"]   for i in range(p["n_lstm_layers"])]
    dropout_per_layer = [p[f"dropout_{i}"] for i in range(p["n_lstm_layers"])]

    best_model = build_lstm(n_features, p["n_lstm_layers"], units_per_layer, dropout_per_layer, p["dense_units"], p["learning_rate"])
    
    best_model.fit(X_cv_s, y_cv_s, epochs=epochs, batch_size=p["batch_size"],
                   callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                   verbose=0)

    testPredict = y_scaler.inverse_transform(best_model.predict(X_test_s))
    testY       = y_scaler.inverse_transform(y_test_s.reshape(-1, 1))
    test_rmse   = np.sqrt(mean_squared_error(testY, testPredict))
    print(f"Test RMSE: {test_rmse:.4f}")

    pd.DataFrame({
        "actual":    testY.flatten(),
        "predicted": testPredict.flatten(),
        "error":     (testY - testPredict).flatten()
    }).to_csv("predictions.csv", index=False)

    study.trials_dataframe().to_csv("optuna_trials.csv", index=False)

    pd.DataFrame([{**study.best_params, "cv_rmse": study.best_value,
                   "test_rmse": test_rmse}]).to_csv("best_params.csv", index=False)

    return {
        "best_params": study.best_params,
        "cv_rmse":     study.best_value,
        "test_rmse":   test_rmse,
        "predictions": testPredict,
        "actuals":     testY,
        "model":       best_model,
        "study":       study
    }