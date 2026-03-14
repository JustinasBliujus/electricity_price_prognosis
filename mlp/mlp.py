import tensorflow as tf
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
Sequential = tf.keras.models.Sequential
Dense      = tf.keras.layers.Dense
Dropout    = tf.keras.layers.Dropout
Input      = tf.keras.layers.Input
from mlp.build_mlp import build_mlp
from sklearn.metrics import mean_squared_error

CATEGORICAL_COLS = ['hour', 'day', 'month', 'year', 'dayofweek', 'quarter', 'dayofyear', 'weekend']
CONTINUOUS_COLS  = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
                    'rolling_mean_6', 'rolling_std_6', 'rolling_mean_12',
                    'rolling_std_12', 'rolling_mean_24', 'rolling_std_24']

def get_col_indices(all_columns, cols_to_scale):
    return [all_columns.index(c) for c in cols_to_scale if c in all_columns]

def mlp(X, y, n_splits=5, test_size=0.2, epochs=100, batch_size=32,
        X_scaler_class = RobustScaler, y_scaler_class = RobustScaler, scale_all = False):

    split_idx = int(len(X) * (1 - test_size))
    X_cv, X_test = X[:split_idx], X[split_idx:]
    y_cv, y_test = y[:split_idx], y[split_idx:]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []

    all_columns = CATEGORICAL_COLS + CONTINUOUS_COLS

    if scale_all:
        cols_to_scale = list(range(X_cv.shape[1]))  
    else:
        cols_to_scale = get_col_indices(all_columns, CONTINUOUS_COLS)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_train, X_val = X_cv[train_idx], X_cv[val_idx]
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]

        X_scaler = X_scaler_class()
        y_scaler = y_scaler_class()

        X_train_scaled = X_train.copy()
        X_val_scaled   = X_val.copy()

        X_train_scaled[:, cols_to_scale] = X_scaler.fit_transform(X_train[:, cols_to_scale])
        X_val_scaled[:, cols_to_scale]   = X_scaler.transform(X_val[:, cols_to_scale])

        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        n_features = X_train_scaled.shape[1]

        model = build_mlp(n_features=n_features,n_mlp_layers=2,dropout_rate=[0.2,0.2],
                          dense_units=[128,64],learning_rate=0.001,activation='relu')

        model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val_scaled),
            verbose=1
        )

        y_val_pred_scaled = model.predict(X_val_scaled).flatten()
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
        val_rmse_orig = np.sqrt(np.mean((y_val_pred - y_val) ** 2))

        fold_rmses.append({"fold": fold + 1, "val_rmse": val_rmse_orig})
        print(f"Fold {fold + 1}, val_rmse: {val_rmse_orig:.4f}")

    X_test_scaled = X_test.copy()
    X_test_scaled[:, cols_to_scale] = X_scaler.transform(X_test[:, cols_to_scale])
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    testPredict = model.predict(X_test_scaled)
    testPredict = y_scaler.inverse_transform(testPredict)
    testY = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

    test_rmse = np.sqrt(mean_squared_error(testY, testPredict))
    print(f"Test RMSE: {test_rmse:.4f}")

    return {
        "fold_rmses": fold_rmses,
        "mean_cv_rmse": np.mean([f["val_rmse"] for f in fold_rmses]),
        "test_rmse": test_rmse,
        "predictions": testPredict,
        "actuals": testY,
        "model": model
    }