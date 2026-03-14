import os
import pandas as pd
from datetime import datetime
from utils.baseline_models import calculate_baseline_models
from lgbm.lgbm import lgbm
from xgb.xgboost import xgb
from mlp.mlp import mlp
from lstm.lstm import lstm

CSV_PATH = "baseline_results.csv"

def get_baselines(X, y, y_train, y_test, X_train, X_test):
    baseline_results = calculate_baseline_models(y_train, y_test, X_test)
    lgbm_results_regression = lgbm(X.to_numpy(), y.to_numpy(), objective='regression')
    lgbm_results_huber = lgbm(X.to_numpy(), y.to_numpy(), objective='huber')
    lgbm_results_fair = lgbm(X.to_numpy(), y.to_numpy(), objective='fair')
    xgb_results_regression = xgb(X.to_numpy(), y.to_numpy(),objective='reg:squarederror')
    xgb_results_huber = xgb(X.to_numpy(), y.to_numpy(),objective='reg:pseudohubererror')
    mlp_results      = mlp(X.to_numpy(), y.to_numpy())
    lstm_results     = lstm(X.to_numpy(), y.to_numpy())

    rows = [
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "last_value",     "rmse": baseline_results["last_value"][1]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "train_mean",     "rmse": baseline_results["train_mean"][1]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "rolling_mean",   "rmse": baseline_results["rolling_mean"][1]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "lgbm_regression","rmse": lgbm_results_regression["test_rmse"], "mean_cv_rmse": lgbm_results_regression["mean_cv_rmse"]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "lgbm_huber",     "rmse": lgbm_results_huber["test_rmse"], "mean_cv_rmse": lgbm_results_huber["mean_cv_rmse"]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "lgbm_fair",      "rmse": lgbm_results_fair["test_rmse"], "mean_cv_rmse": lgbm_results_fair["mean_cv_rmse"]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "xgb_regression", "rmse": xgb_results_regression["test_rmse"],  "mean_cv_rmse": xgb_results_regression["mean_cv_rmse"]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "xgb_huber",      "rmse": xgb_results_huber["test_rmse"],  "mean_cv_rmse": xgb_results_huber["mean_cv_rmse"]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "mlp",            "rmse": mlp_results["test_rmse"], "mean_cv_rmse": mlp_results["mean_cv_rmse"]},
        {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "lstm",           "rmse": lstm_results["test_rmse"],  "mean_cv_rmse": lstm_results["mean_cv_rmse"]},
    ]

    if os.path.exists(CSV_PATH):
        df = pd.concat([pd.read_csv(CSV_PATH), pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)

    df.to_csv(CSV_PATH, index=False)
    print(f"\nResults saved to {CSV_PATH}")

    return {
        "baselines":        baseline_results,
        "lgbm_regression":  lgbm_results_regression,
        "lgbm_huber":       lgbm_results_huber,
        "lgbm_fair":        lgbm_results_fair,
        "xgb_regression":   xgb_results_regression,
        "xgb_huber":        xgb_results_huber,
        "mlp":              mlp_results,
        "lstm":             lstm_results
    }