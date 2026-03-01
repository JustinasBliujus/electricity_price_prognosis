from prepare_data import prepare_data
from get_baselines import get_baselines
from utils.visualization import generate_all_plots
from optimize_lightgbm import optimize_lightgbm
from optimize_xgboost import optimize_xgboost
import pandas as pd

def xgboost_studies(df_clean):

    study_l2, trial_params_l2, trial_value_l2 = optimize_xgboost(
        df_clean, n_trials=150, metric='rmse', n_splits=7, test_size=0.1, 
        loss_function="reg:squarederror"
    )

    study_l1, trial_params_l1, trial_value_l1 = optimize_xgboost(
        df_clean, n_trials=150, metric='rmse', n_splits=7, test_size=0.1, 
        loss_function="reg:absoluteerror"
    )

    study_huber, trial_params_huber, trial_value_huber = optimize_xgboost(
        df_clean, n_trials=150, metric='rmse', n_splits=7, test_size=0.1, 
        loss_function="reg:pseudohubererror"
    )


    print(f"\nREGRESSION (L2)")
    print(f"   Best rmse {trial_value_l2:.4f}")
    print(f"   Best Params: {trial_params_l2}")

    print(f"\nREGRESSION L1 (L1)")
    print(f"   Best rmse: {trial_value_l1:.4f}")
    print(f"   Best Params: {trial_params_l1}")

    print(f"\nHUBER")
    print(f"   Best rmse: {trial_value_huber:.4f}")
    print(f"   Best Params: {trial_params_huber}")

    best_value = min(trial_value_l2, trial_value_l1, trial_value_huber)
    best_loss = {
        trial_value_l2: "REGRESSION (L2)",
        trial_value_l1: "REGRESSION L1",
        trial_value_huber: "HUBER",
    }[best_value]

    print(f"BEST OVERALL: {best_loss} with rmse = {best_value:.4f}")

    results_df = pd.DataFrame([
        {
            'loss_function': 'regression',
            'best_rmse': trial_value_l2,
            **trial_params_l2
        },
        {
            'loss_function': 'regression_l1',
            'best_rmse': trial_value_l1,
            **trial_params_l1
        },
        {
            'loss_function': 'huber',
            'best_rmse': trial_value_huber,
            **trial_params_huber
        },
    ])
    
    csv_path = 'xgboost_optimization_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")