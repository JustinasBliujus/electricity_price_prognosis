from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from lstm.lstm import LSTMModel
from mlp.mlp import MLPModel
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def split(X, y, test_size=0.2):
        split_index = int(len(X) * (1 - test_size))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

global_X_scaler = RobustScaler()
global_y_scaler = RobustScaler()

def train_direct_models(X, y, N_SPLITS, TEST_SIZE, model_class, epochs=None):
    X_train, X_test, y_train, y_test = split(X, y)
    global_X_scaler.fit(X_train.to_numpy())
    global_y_scaler.fit(y_train.values.reshape(-1, 1))
    models = {}
    for hour in range(24):
        print(hour)
        models[hour] = train_direct_model_one_hour(X, y, hour, N_SPLITS, TEST_SIZE, model_class, epochs)
    return models

def split_and_shift_data(X,y,hour):
    X_train, X_test, y_train, y_test = split(X,y)
    
    y_train_shifted = y_train.shift(-hour).dropna()
    X_train_aligned = X_train.iloc[:len(y_train_shifted)] 
    y_train_shifted = y_train_shifted.to_numpy()
    X_train_aligned = X_train_aligned.to_numpy()
    
    return X_train_aligned, y_train_shifted

def train_direct_model_one_hour(X, y, hour, N_SPLITS, TEST_SIZE, model_class, epochs=None):
    X_train, X_test, y_train, y_test = split(X, y)
    global_X_scaler.fit(X_train.to_numpy())
    global_y_scaler.fit(y_train.values.reshape(-1, 1))
    X_train_aligned, y_train_shifted = split_and_shift_data(X,y,hour)
    
    model_params = {"n_splits": N_SPLITS, "test_size": TEST_SIZE}

    if epochs is not None and isinstance(model_class, type) and issubclass(model_class, (MLPModel, LSTMModel)):
        model_params["epochs"] = epochs
        model_params["X_scaler"] = global_X_scaler
        model_params["y_scaler"] = global_y_scaler

    model = model_class(**model_params)

    model.fit_final_24(X_train_aligned, y_train_shifted)
    return model

def make_direct_prediction(models, X_test, y_test, X_train):
    mae_total = 0
    rmse_total = 0
    prediction_history = []
    n_windows = len(X_test) - 24

    for i in range(n_windows):
        current_X = X_test.iloc[i].values.reshape(1, -1)
        next_day_y = y_test.iloc[i:i + 24].values
        
        predictions = []
        for hour in range(24):
            model = models[hour]
            if isinstance(model, LSTMModel):
                time_steps = model.time_steps
                start_index = max(0, i - time_steps + 1)
                X_window = X_test.iloc[start_index:i + 1].values
                if len(X_window) < time_steps:
                    pre_sequence_length = time_steps - len(X_window)
                    pre_sequence = X_train.iloc[-pre_sequence_length:].values
                    X_window = np.concatenate([pre_sequence, X_window], axis=0)
                X_scaled = model.X_scaler.transform(X_window)
                X_scaled = X_scaled.reshape(1, time_steps, -1)
                pred_scaled = model.model(X_scaled, training=False).numpy()
                pred_original = model.y_scaler.inverse_transform(pred_scaled)
                pred = pred_original[0][0]
            elif isinstance(model, MLPModel):
                X_scaled = model.X_scaler.transform(current_X)
                pred_scaled = model.model(X_scaled, training=False).numpy()
                pred_original = model.y_scaler.inverse_transform(pred_scaled)
                pred = pred_original[0][0]
            else:
                pred = model.model.predict(current_X)[0]
            predictions.append(pred)

        mae_total += mean_absolute_error(next_day_y, predictions)
        rmse_total += root_mean_squared_error(next_day_y, predictions)
        prediction_history.append(predictions)

        if i % 100 == 0:
            print(f"window {i}/{n_windows} MAE {mae_total/(i+1)}, RMSE {rmse_total/(i+1)}")

    mae = mae_total / n_windows
    rmse = rmse_total / n_windows
    print(f"\nMAE: {mae}, RMSE: {rmse}")

    pd.DataFrame(prediction_history).to_csv(
        os.path.join(CURRENT_DIR, "direct_predictions.csv"), index=False
    )
    return mae, rmse, prediction_history
