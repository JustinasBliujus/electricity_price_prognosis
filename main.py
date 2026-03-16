from utils.prepare_data import prepare_data
from utils.get_baselines import get_baselines
from utils.visualization import generate_all_plots
from mlp.mlp_test_scalers import mlp_test_scalers
from lgbm.lgbm import lgbm, lgbm_optuna
from xgb.xgboost import xgb, xgboost_optuna
from mlp.mlp import mlp, mlp_optuna
from lstm.lstm import lstm, lstm_optuna

#pip install tensorflow numpy matplotlib lightgbm xgboost pandas optuna scikit-learn seaborn

def main():
    df_clean, X, y, X_train, X_test, y_train, y_test, folder_path_visualizations = prepare_data()

    #generate_all_plots(df_clean, X, y_train, y_test, folder_path_visualizations)
    #mlp(X.to_numpy(),y.to_numpy())
    #lgbm(X.to_numpy(),y.to_numpy(),objective='fair')
    #xgb(X.to_numpy(), y.to_numpy())
    #lstm(X.to_numpy(), y.to_numpy())

    get_baselines(X, y, y_train, y_test, X_train, X_test)

    #mlp_test_scalers(X=X.to_numpy(), y=y.to_numpy(),scale_all=True)
    #mlp_test_scalers(X=X.to_numpy(), y=y.to_numpy(),scale_all=False)

    #lgbm_optuna(X.to_numpy(), y.to_numpy(), n_trials=250)
    #xgboost_optuna(X.to_numpy(), y.to_numpy(), n_trials=250)
    #mlp_optuna(X.to_numpy(), y.to_numpy(), n_trials=100)
    #lstm_optuna(X.to_numpy(), y.to_numpy(), n_trials=100)

if __name__ == "__main__":
    main()