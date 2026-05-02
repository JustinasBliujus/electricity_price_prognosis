from utils.prepare_data import prepare_data
from utils.get_baselines import get_baselines
from utils.visualization import generate_all_plots
from lgbm.lgbm import lgbm_run, lgbm_optuna
from xgb.xgboost import xgb_run, xgb_optuna
from mlp.mlp import mlp_run, mlp_optuna
from lstm.lstm import lstm_run, lstm_optuna
from feature_importance.feature_importance import plot_importance, plot_importance_mlp, plot_importance_lstm
from calculate_results.aggregate import aggregate, compare_relative_mae, plot_from_csv, plot_rmse, plot_mae,plot_baseline_fold2
from calculate_results.error_profiling import get_hourly_test_results, plot_mae_by_feature,plot_all_models
#pip install tensorflow numpy matplotlib lightgbm xgboost pandas optuna scikit-learn seaborn plotly kaleido statsmodels shap

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

N_SPLITS = 5
TEST_SIZE = 0.2
EPOCHS = 100
N_TRIALS = 1000

def main():
    df_clean, X, y, X_train, X_test, y_train, y_test, folder_path_visualizations = prepare_data()
    #residuals = np.abs(y_train - X_train["lag_1"])
    #print(np.percentile(residuals, [25, 50, 75, 90, 95, 99]))
    #[  2.23     8.77    27.82    54.542   76.894  137.4712]
    #graphs_for_report()
    #generate_all_plots(df_clean, X, y_train, y_test, folder_path_visualizations)
    #get_baselines(X, y, y_train, y_test, X_train, X_test)
    results = lstm_run(X.to_numpy(), y.to_numpy(),N_SPLITS,TEST_SIZE,EPOCHS)
    model = results["model"].model
    plot_importance_lstm(model, X.columns.tolist(), X, os.path.dirname(os.path.abspath(__file__)))

    #plot_baseline_fold2(X,y,N_SPLITS,TEST_SIZE)
    #aggregate("lstm")
    #compare_relative_mae()
    #plot_from_csv()
    #plot_rmse()
    #plot_mae()
    #plot_all_models(X)
    #TODO do t+24 models + ensemble
    #ataskaitoj:
    # 2 foldo analize ?
    # rmse ir mae atvirksiai, kodel?
    # vaizdai pagal valadnas ir dienas
    # shap analize
if __name__ == "__main__":
    main()