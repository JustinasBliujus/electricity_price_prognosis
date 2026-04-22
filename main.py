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

N_SPLITS = 5
TEST_SIZE = 0.2
EPOCHS = 100
N_TRIALS = 1000

def main():
    df_clean, X, y, X_train, X_test, y_train, y_test, folder_path_visualizations = prepare_data()
    #y_pred_last = y_test.shift(1).fillna(y_train.iloc[-1])
    #residuals = np.abs(y_test - y_pred_last)
    #print(np.percentile(residuals, [25, 50, 75, 90, 95, 99]))
    #[  2.23     8.77    27.82    54.542   76.894  137.4712]

    #generate_all_plots(df_clean, X, y_train, y_test, folder_path_visualizations)
    #get_baselines(X, y, y_train, y_test, X_train, X_test)

    #lgbm_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE)
    #lgbm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, n_trials=N_TRIALS, objective="fair")
    #plot_optuna_results("lgbm/lgbm/optuna_results/optuna_trials.csv")

    #xgb_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE)
    xgb_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, n_trials=N_TRIALS,objective="reg:fair")

    #mlp_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS)
    #mlp_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS,activation="sigmoid")
    #mlp_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS,activation="linear")
    
    #lstm_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS)
    #lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="tanh")
    #lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="sigmoid")
    #lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS, activation="linear")

    #TODO make xgboost custom fair
    #TODO run sigmoid linear on mlp
    #TODO run lstm on tanh sigmoid linear

    #TODO then having best models
    #
    #TODO CV mse, mse mae rMae on test?
    
    #TODO do feature importance analysis SHAP or inner models feature importance
    #TODO check when do models fail to prognose, hours,days...
    #TODO model ensemble
if __name__ == "__main__":
    main()