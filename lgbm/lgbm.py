from time_series_models.tree_time_series_model import TreeTimeSeriesModel
import lightgbm as lgbm
import os
from datetime import datetime

class LGBMModel(TreeTimeSeriesModel):
    def __init__(self,
                 objective='regression',
                 n_estimators=500,
                 learning_rate=0.01,
                 max_depth=6,
                 num_leaves=32,
                 feature_fraction=0.8,
                 min_child_samples=500,
                 alpha=15,
                 fair_c=7,
                 n_splits=None,
                 test_size=None,
                 baseline=27.586892100211507,
                 output_dir=None):

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"lgbm_{date_str}"
            )
            
        os.makedirs(output_dir, exist_ok=True)

        super().__init__(n_splits=n_splits, test_size=test_size,
                         output_dir=output_dir, baseline=baseline)

        self.objective         = objective
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.num_leaves        = num_leaves
        self.feature_fraction  = feature_fraction
        self.min_child_samples = min_child_samples
        self.alpha             = alpha
        self.fair_c            = fair_c

    def build_model(self, n_features=None):
        return self.build_lightgbm()

    def build_lightgbm(self):
        return lgbm.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            min_child_samples=self.min_child_samples,
            objective=self.objective,
            alpha=self.alpha,
            fair_c=self.fair_c,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    def suggest_hyperparams(self, trial):
        self.n_estimators      = trial.suggest_int("n_estimators", 100, 1000)
        self.learning_rate     = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
        self.max_depth         = trial.suggest_int("max_depth", 3, 10)
        self.num_leaves        = trial.suggest_int("num_leaves", 15, 110)
        self.feature_fraction  = trial.suggest_float("feature_fraction", 0.4, 1.0)
        self.min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
        if self.objective == "huber":
            self.alpha  = trial.suggest_float("alpha", 2.0, 60.0, log=True)
        if self.objective == "fair":
            self.fair_c = trial.suggest_float("fair_c", 2.0, 60.0, log=True)

    def apply_best_params(self, p):
        self.n_estimators      = p["n_estimators"]
        self.learning_rate     = p["learning_rate"]
        self.max_depth         = p["max_depth"]
        self.num_leaves        = p["num_leaves"]
        self.feature_fraction  = p["feature_fraction"]
        self.min_child_samples = p["min_child_samples"]
        if self.objective == "huber":
            self.alpha  = p["alpha"]
        if self.objective == "fair":
            self.fair_c = p["fair_c"]

def lgbm_run(X=None, y=None, n_splits=None, test_size=None, objective='regression'):
    model = LGBMModel(n_splits=n_splits, test_size=test_size, objective=objective)
    print(model.__dict__)
    return model.run(X, y)
    

def lgbm_optuna(X, y, n_splits=None, test_size=None, n_trials=None, objective='regression'):
    return LGBMModel(n_splits=n_splits,
                     test_size=test_size, objective=objective).run_optuna(X, y, n_trials=n_trials)