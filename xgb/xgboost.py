from time_series_models.tree_time_series_model import TreeTimeSeriesModel
import xgboost
import os
import numpy as np
from datetime import datetime

class XGBModel(TreeTimeSeriesModel):
    def __init__(self,
                 objective='reg:squarederror',
                 n_estimators=500,
                 learning_rate=0.01,
                 max_depth=6,
                 colsample_bytree=0.8,
                 min_child_weight=1,
                 huber_slope=4,
                 fair_slope=1,
                 n_splits=None,
                 test_size=None,
                 baseline=35.97867655711516,
                 output_dir=None):
        
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"xgb_{date_str}"
            )

        os.makedirs(output_dir, exist_ok=True)

        super().__init__(n_splits=n_splits, test_size=test_size,
                         output_dir=output_dir, baseline=baseline)

        self.objective        = objective
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.huber_slope      = huber_slope
        self.fair_slope       = fair_slope

    def build_model(self, n_features=None):
        return self.build_xgboost()

    def build_xgboost(self):
        use_fair = self.objective == "reg:fair"
        c = self.fair_slope

        params = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=42,
            n_jobs=-1,
        )

        if use_fair:
            params["objective"] = lambda y_true, y_pred: fair_objective(y_true, y_pred, c=c)
        else:
            params["objective"] = self.objective
            if self.objective == "reg:pseudohubererror":
                params["huber_slope"] = self.huber_slope

        return xgboost.XGBRegressor(**params)

    def suggest_hyperparams(self, trial):
        self.n_estimators     = trial.suggest_int("n_estimators", 100, 1500, step=100)
        self.learning_rate    = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
        self.max_depth        = trial.suggest_int("max_depth", 3, 10)
        self.colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        self.min_child_weight = trial.suggest_int("min_child_weight", 1, 100)
        if(self.objective == "reg:pseudohubererror"):
            self.huber_slope      = trial.suggest_float("huber_slope", 25, 110, log=True)
        if self.objective == "reg:fair":
            self.fair_slope   = trial.suggest_float("fair_slope", 25, 110, log=True)

    def apply_best_params(self, p):
        self.n_estimators     = p["n_estimators"]
        self.learning_rate    = p["learning_rate"]
        self.max_depth        = p["max_depth"]
        self.colsample_bytree = p["colsample_bytree"]
        self.min_child_weight = p["min_child_weight"]
        if self.objective == "reg:pseudohubererror":
            self.huber_slope      = p["huber_slope"]
        if self.objective == "reg:fair":
            self.fair_slope   = p.get("fair_slope", self.fair_slope)

def xgb_run(X=None, y=None, n_splits=None, test_size=None, objective="reg:squarederror"):
    model = XGBModel(n_splits=n_splits, test_size=test_size, objective=objective)
    print(model.__dict__)
    return model.run(X, y)

def xgb_optuna(X, y, n_splits=None, test_size=None, n_trials=None, objective="reg:squarederror"):
    return XGBModel(n_splits=n_splits,test_size=test_size, objective=objective).run_optuna(X, y, n_trials=n_trials)


def fair_objective(y_true, y_pred, c):
    x = y_pred - y_true
    den = np.abs(x) + c
    grad = c * x / den
    hess = c * c / (den ** 2)
    return grad, hess