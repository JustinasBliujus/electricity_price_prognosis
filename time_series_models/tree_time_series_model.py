from .base_time_series_model import BaseTimeSeriesModel
import numpy as np
import warnings

class TreeTimeSeriesModel(BaseTimeSeriesModel):

    def fit_fold(self, X_train, y_train, X_val, y_val):
        model   = self.build_model()
        model.fit(X_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pred = model.predict(X_val)
        return pred, y_val

    def fit_final(self, X_cv, y_cv, X_test, y_test):
        model  = self.build_model()
        model.fit(X_cv, y_cv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pred = model.predict(X_test)
        return pred, y_test