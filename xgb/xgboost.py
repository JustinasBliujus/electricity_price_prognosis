from time_series_models.tree_time_series_model import TreeTimeSeriesModel
import xgboost
import os

class XGBModel(TreeTimeSeriesModel):
    def __init__(self,
                 objective='reg:squarederror',
                 n_estimators=500,
                 learning_rate=0.01,
                 max_depth=6,
                 colsample_bytree=0.8,
                 min_child_weight=1,
                 subsample=1.0,
                 huber_slope=4,
                 n_splits=None,
                 test_size=None,
                 baseline=25.806314985359016,
                 output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb")):

        os.makedirs(output_dir, exist_ok=True)

        super().__init__(n_splits=n_splits, test_size=test_size,
                         output_dir=output_dir, baseline=baseline)

        self.objective        = objective
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.subsample        = subsample
        self.huber_slope      = huber_slope

    def build_model(self, n_features=None):
        return self.build_xgboost()

    def build_xgboost(self):
        return xgboost.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            objective=self.objective,
            huber_slope=self.huber_slope,
            random_state=42,
            n_jobs=-1,
        )

    def suggest_hyperparams(self, trial):
        self.n_estimators     = trial.suggest_int("n_estimators", 100, 1000, step=100)
        self.learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        self.max_depth        = trial.suggest_int("max_depth", 3, 10)
        self.colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        self.min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        self.subsample        = trial.suggest_float("subsample", 0.5, 1.0)

    def apply_best_params(self, p):
        self.n_estimators     = p["n_estimators"]
        self.learning_rate    = p["learning_rate"]
        self.max_depth        = p["max_depth"]
        self.colsample_bytree = p["colsample_bytree"]
        self.min_child_weight = p["min_child_weight"]
        self.subsample        = p["subsample"]

def xgb_run(X=None, y=None, n_splits=None, test_size=None, objective="reg:squarederror"):
    model = XGBModel(n_splits=n_splits, test_size=test_size, objective=objective)
    print(model.__dict__)
    return model.run(X, y)

def xgb_optuna(X, y, n_splits=None, test_size=None, n_trials=None):
    return XGBModel(n_splits=n_splits,
                    test_size=test_size).run_optuna(X, y, n_trials=n_trials)