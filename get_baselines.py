from utils.visualization import generate_all_plots
from utils.baseline_models import calculate_baseline_models
from utils.build_default_lightgbm_model import build_default_lightgbm_model
from utils.build_default_xgboost_model import build_default_xgboost_model

def get_baselines(y_train, y_test, X_train, X_test):
    calculate_baseline_models(y_train, y_test, X_test)
    build_default_lightgbm_model(X_train, y_train, X_test, y_test)
    build_default_xgboost_model(X_train, y_train, X_test, y_test)