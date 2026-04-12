from utils.prepare_data import prepare_data
from utils.get_baselines import get_baselines
from utils.visualization import generate_all_plots
from lgbm.lgbm import lgbm_run, lgbm_optuna
from xgb.xgboost import xgb_run, xgb_optuna
from mlp.mlp import mlp_run, mlp_optuna
from lstm.lstm import lstm_run, lstm_optuna

#pip install tensorflow numpy matplotlib lightgbm xgboost pandas optuna scikit-learn seaborn plotly kaleido statsmodels

N_SPLITS = 5
TEST_SIZE = 0.2
EPOCHS = 100
N_TRIALS = 10

def main():
    df_clean, X, y, X_train, X_test, y_train, y_test, folder_path_visualizations = prepare_data()
    #generate_all_plots(df_clean, X, y_train, y_test, folder_path_visualizations)
    #get_baselines(X, y, y_train, y_test, X_train, X_test)

    #lgbm_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE)
    #lgbm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, n_trials=N_TRIALS)

    #xgb_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE)
    xgb_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, n_trials=N_TRIALS)

    #mlp_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS)
    mlp_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS)

    #lstm_run(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS)
    lstm_optuna(X=X.to_numpy(), y=y.to_numpy(), n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS, n_trials=N_TRIALS)

if __name__ == "__main__":
    main()