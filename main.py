from prepare_data import prepare_data
from get_baselines import get_baselines
from utils.visualization import generate_all_plots

def main():
    df_clean, X, y, X_train, X_test, y_train, y_test, folder_path_visualizations = prepare_data()

    #generate_all_plots(df_clean, X, y_train, y_test, folder_path_visualizations)

    get_baselines(y_train, y_test, X_train, X_test)

    # normalizing comparison
    # optuna light gbm, fair, huber
    # use tiemseriesplit for validation

if __name__ == "__main__":
    main()