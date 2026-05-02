import tensorflow as tf
from time_series_models.neural_time_series_model import NeuralTimeSeriesModel
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import os
from datetime import datetime
import numpy as np

EarlyStopping = tf.keras.callbacks.EarlyStopping
Sequential    = tf.keras.models.Sequential
Dense         = tf.keras.layers.Dense
Dropout       = tf.keras.layers.Dropout
Input         = tf.keras.layers.Input
LSTM          = tf.keras.layers.LSTM

class LSTMModel(NeuralTimeSeriesModel):
    def __init__(self,
                 n_lstm_layers=1,
                 units_per_layer=[32],
                 dropout_per_layer=[0.2],
                 dense_units=32,
                 learning_rate=0.001,
                 X_scaler_class=RobustScaler,
                 y_scaler_class=RobustScaler,
                 batch_size=32,
                 epochs=None,
                 n_splits=None,
                 test_size=None,
                 activation=None,
                 baseline=17.120583933251048,
                 output_dir=None,
                 time_steps=1):

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"lstm_{date_str}"
            )

        os.makedirs(output_dir, exist_ok=True)

        super().__init__(X_scaler_class=X_scaler_class, y_scaler_class=y_scaler_class,
                         epochs=epochs, batch_size=batch_size, n_splits=n_splits,
                         test_size=test_size, output_dir=output_dir, baseline=baseline)

        self.n_lstm_layers     = n_lstm_layers
        self.units_per_layer   = units_per_layer
        self.dropout_per_layer = dropout_per_layer
        self.dense_units       = dense_units
        self.learning_rate     = learning_rate
        self.activation        = activation
        self.time_steps        = time_steps 
        
    def build_model(self, n_features):
        return self.build_lstm(n_features)

    def build_lstm(self, n_features):
        layers = [Input(shape=(self.time_steps, n_features))]
        for i in range(self.n_lstm_layers):
            return_seq = (i < self.n_lstm_layers - 1)
            layers.append(LSTM(self.units_per_layer[i], return_sequences=return_seq))
            layers.append(Dropout(self.dropout_per_layer[i]))

        layers.append(Dense(self.dense_units, activation=self.activation))
        layers.append(Dense(1))

        model = Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error"
        )
        return model
    
    def create_sequences(self, X, y):
        n = len(X) - self.time_steps + 1
        idx = np.arange(self.time_steps)[None, :] + np.arange(n)[:, None]
        return X[idx], y[idx[:, -1]]

    def fit_fold(self, X_train, y_train, X_val, y_val):
        X_scaler = self.X_scaler_class()
        y_scaler = self.y_scaler_class()

        X_train_s = X_scaler.fit_transform(X_train)
        X_val_s   = X_scaler.transform(X_val)

        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        X_train_s, y_train_s = self.create_sequences(X_train_s, y_train_s)
        X_val_s,   y_val_s   = self.create_sequences(X_val_s,   y_val_s)

        self.y_scaler = y_scaler
        self.X_scaler = X_scaler

        model = self.build_model(X_train_s.shape[-1])
        model.fit(X_train_s, y_train_s, epochs=self.epochs, batch_size=self.batch_size,
                validation_data=(X_val_s, y_val_s),
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0)

        pred = y_scaler.inverse_transform(model.predict(X_val_s))
        return pred, y_val[self.time_steps - 1:]

    def fit_final(self, X_cv, y_cv, X_test, y_test):
        X_scaler = self.X_scaler_class()
        y_scaler = self.y_scaler_class()

        X_cv_s   = X_scaler.fit_transform(X_cv)
        X_test_s = X_scaler.transform(X_test)

        y_cv_s = y_scaler.fit_transform(y_cv.reshape(-1, 1)).flatten()

        X_cv_s,  y_cv_s = self.create_sequences(X_cv_s, y_cv_s)
        X_test_s, _     = self.create_sequences(X_test_s, y_test)

        model = self.build_model(X_cv_s.shape[-1])
        model.fit(X_cv_s, y_cv_s, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        pred = y_scaler.inverse_transform(model.predict(X_test_s))
        return pred, y_test[self.time_steps - 1:]
    
    
    def suggest_hyperparams(self, trial):
        self.n_lstm_layers     = trial.suggest_int("n_lstm_layers", 1, 2)
        self.batch_size        = trial.suggest_categorical("batch_size", [16, 32, 64])
        self.learning_rate     = trial.suggest_float("learning_rate", 0.001, 0.03, log=True)
        self.dense_units       = trial.suggest_int("dense_units", 16, 128, step=16)
        self.units_per_layer   = [trial.suggest_int(f"units_{i}", 32, 256, step=32) for i in range(self.n_lstm_layers)]
        self.dropout_per_layer = [trial.suggest_float(f"dropout_{i}", 0.1, 0.5) for i in range(self.n_lstm_layers)]
        self.time_steps = trial.suggest_categorical("time_steps", [1,6,12,24])

    def apply_best_params(self, p):
        self.n_lstm_layers     = p["n_lstm_layers"]
        self.learning_rate     = p["learning_rate"]
        self.dense_units       = p["dense_units"]
        self.batch_size        = p["batch_size"]
        self.units_per_layer   = [p[f"units_{i}"] for i in range(p["n_lstm_layers"])]
        self.dropout_per_layer = [p[f"dropout_{i}"] for i in range(p["n_lstm_layers"])]
        self.time_steps = p["time_steps"]

def lstm_run(X=None, y=None, n_splits=None, test_size=None, epochs=None, activation='relu', time_steps=168):
    model = LSTMModel(epochs=epochs, n_splits=n_splits, test_size=test_size, activation=activation, time_steps=time_steps)
    return model.run(X, y)

def lstm_optuna(X, y, n_splits=None, test_size=None, epochs=None, n_trials=None, activation='relu'):
    return LSTMModel(epochs=epochs, n_splits=n_splits,
                     test_size=test_size, activation=activation).run_optuna(X, y, n_trials=n_trials)