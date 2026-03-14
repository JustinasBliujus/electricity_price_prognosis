from utils.prepare_data import prepare_data
from utils.get_baselines import get_baselines
from utils.visualization import generate_all_plots
import pandas as pd
from mlp.mlp_test_scalers import mlp_test_scalers

#pip install tensorflow numpy matplotlib lightgbm xgboost pandas optuna scikit-learn seaborn
def main():
    df_clean, X, y, X_train, X_test, y_train, y_test, folder_path_visualizations = prepare_data()

    #generate_all_plots(df_clean, X, y_train, y_test, folder_path_visualizations)

    get_baselines(X, y, y_train, y_test, X_train, X_test)

    #mlp_test_scalers(X=X.to_numpy(), y=y.to_numpy(),scale_all=True)
    #mlp_test_scalers(X=X.to_numpy(), y=y.to_numpy(),scale_all=False)


if __name__ == "__main__":
    main()