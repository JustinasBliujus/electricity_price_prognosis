from .base_time_series_model import BaseTimeSeriesModel
import tensorflow as tf

EarlyStopping = tf.keras.callbacks.EarlyStopping

class NeuralTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self, 
                 X_scaler_class=None,
                 y_scaler_class=None,
                 epochs=None,
                 batch_size=None,
                 n_splits=None,
                 test_size=None, 
                 output_dir=None,
                 baseline=None):
        
        super().__init__(feature_names=None,
                         n_splits=n_splits, 
                         test_size=test_size,
                         output_dir=output_dir,
                         baseline=baseline)
        
        self.X_scaler_class = X_scaler_class
        self.y_scaler_class = y_scaler_class
        self.epochs         = epochs
        self.batch_size     = batch_size

    def fit_fold(self, X_train, y_train, X_val, y_val):
        X_scaler = self.X_scaler_class()
        y_scaler = self.y_scaler_class()

        X_train_s = X_scaler.fit_transform(X_train)
        X_val_s   = X_scaler.transform(X_val)

        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        self.y_scaler  = y_scaler
        self.X_scaler  = X_scaler

        n_features = X_train_s.shape[-1]
        model = self.build_model(n_features)
        model.fit(X_train_s, y_train_s, epochs=self.epochs, batch_size=self.batch_size,
                  validation_data=(X_val_s, y_val_s),
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=0)

        pred = y_scaler.inverse_transform(model.predict(X_val_s))
        return pred, y_val

    def fit_final(self, X_cv, y_cv, X_test, y_test):
        X_scaler = self.X_scaler_class()
        y_scaler = self.y_scaler_class()

        X_cv_s   = self.reshape_input(X_scaler.fit_transform(X_cv))
        X_test_s = self.reshape_input(X_scaler.transform(X_test))
        y_cv_s   = y_scaler.fit_transform(y_cv.reshape(-1, 1)).flatten()

        model = self.build_model(X_cv_s.shape[-1])
        model.fit(X_cv_s, y_cv_s, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        pred = y_scaler.inverse_transform(model.predict(X_test_s))
        return pred, y_test