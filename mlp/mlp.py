import tensorflow as tf
from time_series_models.neural_time_series_model import NeuralTimeSeriesModel
from sklearn.preprocessing import RobustScaler, MinMaxScaler
EarlyStopping = tf.keras.callbacks.EarlyStopping
import os
from datetime import datetime

Sequential = tf.keras.models.Sequential
Dense      = tf.keras.layers.Dense
Dropout    = tf.keras.layers.Dropout
Input      = tf.keras.layers.Input

class MLPModel(NeuralTimeSeriesModel):
    def __init__(self, 
                 n_mlp_layers=1,
                 units_per_layer=[128],
                 dropout_per_layer=[0.2836132491974548],
                 learning_rate=0.0012584314732730006,
                 activation='relu',
                 X_scaler_class=RobustScaler,
                 y_scaler_class=RobustScaler,
                 batch_size=32,
                 epochs=None,
                 n_splits=None,
                 test_size=None,
                 baseline=36.09601750524018,
                 output_dir=None):

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"mlp_{date_str}"
            )

        os.makedirs(output_dir, exist_ok=True)

        super().__init__(X_scaler_class=X_scaler_class, y_scaler_class=y_scaler_class,
                 epochs=epochs, batch_size=batch_size,n_splits=n_splits, test_size=test_size, output_dir=output_dir, baseline=baseline)
        
        self.n_mlp_layers     = n_mlp_layers
        self.units_per_layer  = units_per_layer
        self.dropout_per_layer = dropout_per_layer
        self.learning_rate    = learning_rate
        self.activation       = activation

    def build_model(self, n_features):
        return self.build_mlp(n_features)

    def build_mlp(self,n_features):
        layers = [Input(shape=(n_features,))]
        for i in range(self.n_mlp_layers):
            layers.append(Dense(self.units_per_layer[i], activation=self.activation))
            layers.append(Dropout(self.dropout_per_layer[i]))
        layers.append(Dense(1))
    
        model = Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
        )
        return model

    def suggest_hyperparams(self, trial):
        self.n_mlp_layers      = trial.suggest_int("n_mlp_layers", 1, 3)
        self.batch_size        = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        self.learning_rate     = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
        self.units_per_layer   = [trial.suggest_int(f"units_{i}", 32, 256, step=32) for i in range(self.n_mlp_layers)]
        self.dropout_per_layer = [trial.suggest_float(f"dropout_{i}", 0.0, 0.5) for i in range(self.n_mlp_layers)]

    def apply_best_params(self, p):
        self.n_mlp_layers      = p["n_mlp_layers"]
        self.learning_rate     = p["learning_rate"]
        self.batch_size        = p["batch_size"]
        self.units_per_layer   = [p[f"units_{i}"] for i in range(p["n_mlp_layers"])]
        self.dropout_per_layer = [p[f"dropout_{i}"] for i in range(p["n_mlp_layers"])]

def mlp_run(X=None, y=None, n_splits=None, test_size=None, epochs=None):
    model = MLPModel(epochs=epochs,n_splits=n_splits,test_size=test_size)
    result = model.run(X,y)
    model.plot_fold_predictions(X, y, fold_number=2)
    return result

def mlp_optuna(X, y, n_splits=None, test_size=None, epochs=None, n_trials=None, activation="relu"):
    return MLPModel(epochs=epochs, n_splits=n_splits,
                    test_size=test_size, activation=activation).run_optuna(X, y, n_trials=n_trials)