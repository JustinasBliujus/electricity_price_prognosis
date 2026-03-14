from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from .build_lstm import build_lstm
Sequential = tf.keras.models.Sequential
Dense      = tf.keras.layers.Dense
Dropout    = tf.keras.layers.Dropout
Input      = tf.keras.layers.Input
LSTM       = tf.keras.layers.LSTM

def lstm(X, y, n_splits=5, test_size=0.2, epochs=100,batch_size=32):
    split_idx = int(len(X) * (1 - test_size))
    X_cv, X_test = X[:split_idx], X[split_idx:]
    y_cv, y_test = y[:split_idx], y[split_idx:]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_train, X_val = X_cv[train_idx], X_cv[val_idx]
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]

        X_scaler = RobustScaler()
        y_scaler = RobustScaler()

        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        X_val_scaled = X_scaler.transform(X_val)
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        n_features = X_train_scaled.shape[1]

        # samples, timesteps, features
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, n_features))
        X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, n_features))
        
        model = build_lstm(n_features=n_features, n_lstm_layers=2, units_per_layer=[128,64],
                   dropout_per_layer=[0.2,0.2], dense_units=32, learning_rate=0.001)

        model.fit(X_train_scaled, y_train_scaled,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(X_val_scaled, y_val_scaled),
                  verbose=1)
        
        valPredict = model.predict(X_val_scaled)
        valPredict = y_scaler.inverse_transform(valPredict)
        valY = y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1))

        fold_rmse = np.sqrt(mean_squared_error(valY, valPredict))
        fold_rmses.append(fold_rmse)
        print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, n_features))

    testPredict = model.predict(X_test_scaled)
    testPredict = y_scaler.inverse_transform(testPredict)
    testY = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

    test_rmse = np.sqrt(mean_squared_error(testY, testPredict))
    print(f"Test RMSE: {test_rmse:.4f}")

    return {
        "fold_rmses": fold_rmses,
        "mean_cv_rmse": np.mean(fold_rmses),
        "test_rmse": test_rmse,
        "predictions": testPredict,
        "actuals": testY,
        "model": model
    }