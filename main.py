from utils.prepare_data import prepare_data
import pandas as pd
from lgbm.lgbm import lgbm_run, lgbm_optuna, LGBMModel
from xgb.xgboost import xgb_run, xgb_optuna
from mlp.mlp import mlp_run, mlp_optuna
from lstm.lstm import lstm_run, lstm_optuna
from feature_importance.feature_importance import plot_importance, plot_importance_lstm, plot_importance_mlp
from additional_data.combine_all import combine_all_datasets
from utils.get_baselines import get_baselines
from utils.visualization import generate_all_plots
from t24.direct import train_direct_models
from t24.t24_baseline import get_baseline
from t24.plotter import plot_direct_predictions_of_windows_all, plot_direct_predictions_by_hour_of_day_all
import os


N_SPLITS = 5 
TEST_SIZE = 0.2
EPOCHS = 100
N_TRIALS = 1000
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
CURRENT_DIR = current_dir = os.path.dirname(os.path.abspath(__file__))
#pip install tensorflow numpy matplotlib lightgbm xgboost pandas optuna scikit-learn seaborn plotly kaleido statsmodels shap

def main():
    df_clean, X, y, X_train, X_test, y_train, y_test = prepare_data()
    
    results = lgbm_run(X,y,N_SPLITS,TEST_SIZE)
    plot_importance(results["model"].model,X.columns,X,"lgbm")
    
if __name__ == "__main__":
    main()