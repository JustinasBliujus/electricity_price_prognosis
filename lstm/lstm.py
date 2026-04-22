import tensorflow as tf
from time_series_models.neural_time_series_model import NeuralTimeSeriesModel
from sklearn.preprocessing import RobustScaler
import os

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
                 baseline=17.120583933251048,
                 output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm")):

        os.makedirs(output_dir, exist_ok=True)

        super().__init__(X_scaler_class=X_scaler_class, y_scaler_class=y_scaler_class,
                         epochs=epochs, batch_size=batch_size, n_splits=n_splits,
                         test_size=test_size, output_dir=output_dir, baseline=baseline)

        self.n_lstm_layers     = n_lstm_layers
        self.units_per_layer   = units_per_layer
        self.dropout_per_layer = dropout_per_layer
        self.dense_units       = dense_units
        self.learning_rate     = learning_rate

    def build_model(self, n_features):
        return self.build_lstm(n_features)

    def build_lstm(self, n_features):
        layers = [Input(shape=(1, n_features))]
        for i in range(self.n_lstm_layers):
            return_seq = (i < self.n_lstm_layers - 1)
            layers.append(LSTM(self.units_per_layer[i], return_sequences=return_seq))
            layers.append(Dropout(self.dropout_per_layer[i]))

        layers.append(Dense(self.dense_units, activation="relu"))
        layers.append(Dense(1))

        model = Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mean_squared_error"
        )
        return model

    def reshape_input(self, X):
        return X.reshape((X.shape[0], 1, X.shape[1]))

    def suggest_hyperparams(self, trial):
        self.n_lstm_layers     = trial.suggest_int("n_lstm_layers", 1, 3)
        self.batch_size        = trial.suggest_categorical("batch_size", [16, 32, 64])
        self.learning_rate     = trial.suggest_float("learning_rate", 0.0001, 0.3, log=True)
        self.dense_units       = trial.suggest_int("dense_units", 16, 128, step=16)
        self.units_per_layer   = [trial.suggest_int(f"units_{i}", 32, 256, step=32) for i in range(self.n_lstm_layers)]
        self.dropout_per_layer = [trial.suggest_float(f"dropout_{i}", 0.1, 0.6) for i in range(self.n_lstm_layers)]

    def apply_best_params(self, p):
        self.n_lstm_layers     = p["n_lstm_layers"]
        self.learning_rate     = p["learning_rate"]
        self.dense_units       = p["dense_units"]
        self.batch_size        = p["batch_size"]
        self.units_per_layer   = [p[f"units_{i}"] for i in range(p["n_lstm_layers"])]
        self.dropout_per_layer = [p[f"dropout_{i}"] for i in range(p["n_lstm_layers"])]

def lstm_run(X=None, y=None, n_splits=None, test_size=None, epochs=None):
    model = LSTMModel(epochs=epochs, n_splits=n_splits, test_size=test_size)
    return model.run(X, y)

def lstm_optuna(X, y, n_splits=None, test_size=None, epochs=None, n_trials=None):
    return LSTMModel(epochs=epochs, n_splits=n_splits,
                     test_size=test_size).run_optuna(X, y, n_trials=n_trials)