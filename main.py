from prepare_data import prepare_data
from get_baselines import get_baselines
from utils.visualization import generate_all_plots
from optimize_lightgbm import optimize_lightgbm
from optimize_xgboost import optimize_xgboost
from lightgbm_studies import lightgbm_studies
from xgboost_studies import xgboost_studies
import pandas as pd
from mlp import run_mlp_with_optimization

def main():
    df_clean, X, y, X_train, X_test, y_train, y_test, X_val, y_val, folder_path_visualizations = prepare_data()

    #generate_all_plots(df_clean, X, y_train, y_test, X_val, y_val, folder_path_visualizations)

    #get_baselines(y_train, y_test, X_train, X_test)

    #lightgbm_studies(df_clean)
    xgboost_studies(df_clean)

if __name__ == "__main__":
    main()