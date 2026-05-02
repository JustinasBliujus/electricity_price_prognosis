from utils.prepare_data import prepare_data
from utils.get_baselines import get_baselines
from utils.visualization import generate_all_plots
from lgbm.lgbm import lgbm_run, lgbm_optuna
from xgb.xgboost import xgb_run, xgb_optuna
from mlp.mlp import mlp_run, mlp_optuna
from lstm.lstm import lstm_run, lstm_optuna

#pip install tensorflow numpy matplotlib lightgbm xgboost pandas optuna scikit-learn seaborn plotly kaleido statsmodels

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
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

    #generate_all_plots(df_clean, X, y_train, y_test, folder_path_visualizations)
    #get_baselines(X, y, y_train, y_test, X_train, X_test)

    #mlp_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS,activation="tanh")
    #mlp_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS,activation="sigmoid")
    #lgbm_run(X.to_numpy(),y.to_numpy(),N_SPLITS,TEST_SIZE)
    #lstm_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS)
    
    #lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="relu")
    #lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="linear")
    #lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="tanh")
    lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="sigmoid")
    #lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="sigmoid") 

    #TODO LSTM time step?
    #
    #TODO CV mse, mse mae rMae on test?
    
    #TODO do feature importance analysis SHAP or inner models feature importance
    #TODO check when do models fail to prognose, hours,days...
    #TODO model ensemble
    
    #TODO do t+24 models + ensemble
if __name__ == "__main__":
    main()