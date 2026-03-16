from sklearn.preprocessing import RobustScaler
from .base_time_series_model import BaseTimeSeriesModel

class NeuralTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self, X_scaler_class=RobustScaler, y_scaler_class=RobustScaler,
                 scale_all=True, epochs=100, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.X_scaler_class = X_scaler_class
        self.y_scaler_class = y_scaler_class
        self.scale_all      = scale_all
        self.epochs         = epochs
        self.batch_size     = batch_size

    def fit_fold(self, model, X_train, y_train, X_val, y_val):
        X_scaler = self.X_scaler_class()
        y_scaler = self.y_scaler_class()

        X_train_s = X_scaler.fit_transform(X_train)
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_val_s   = X_scaler.transform(X_val)
        y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        self.y_scaler = y_scaler

        model.fit(X_train_s, y_train_s, epochs=self.epochs,
                  batch_size=self.batch_size,
                  validation_data=(X_val_s, y_val_s), verbose=0)

        pred = self.y_scaler.inverse_transform(model.predict(X_val_s))
        return pred, y_val