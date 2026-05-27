import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np 
import os
import pandas as pd
from plot_style import PlotStyle

STYLE = PlotStyle()
CURRENT_DIR = base_dir = os.path.dirname(__file__)
MODEL_NAMES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "mlp": "Daugiasluoksnis perceptronas",
    "lstm": "Ilgos trumpalaikės atminties tinklas"
}

def get_models_hourly_preds_by_window(prediction_history,y_test):
    y_test = np.array(y_test)
    y_test_windows = np.array([
        y_test[i:i+24]
        for i in range(len(prediction_history))
    ])

    rmse_by_hour = []
    mae_by_hour = []
    
    for hour in range(24):
        preds_h = prediction_history[str(hour)].values
        actual_h = y_test_windows[:, hour]

        rmse_by_hour.append(root_mean_squared_error(actual_h, preds_h))
        mae_by_hour.append(mean_absolute_error(actual_h, preds_h))
    
    return rmse_by_hour, mae_by_hour

def plot_direct_predictions_of_windows(predictions_folder, y_test):
    path = os.path.join(CURRENT_DIR, predictions_folder, "direct_predictions.csv")
    prediction_history = pd.read_csv(path)
    
    rmse_by_hour, mae_by_hour = get_models_hourly_preds_by_window(prediction_history,y_test)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(24), rmse_by_hour, label=STYLE.get_rmse_label_name())
    ax.plot(range(24), mae_by_hour, label=STYLE.get_mae_label_name())
    ax.set_xticks(range(24))
    ax.set_xlabel('Prognozės lango valanda',fontsize=STYLE.label_size)
    ax.set_ylabel('Paklaida (EUR/MWh)',fontsize=STYLE.label_size)
    STYLE.apply(fig, ax)
    output = os.path.join(CURRENT_DIR, 'direct_rmse_mae_by_window.png')
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    print(output)

def plot_direct_predictions_of_windows_all(y_test):
    all_models = {}
    for folder in ["lgbm", "xgb", "mlp", "lstm"]:
        path = os.path.join(CURRENT_DIR, folder, "direct_predictions.csv")
        if not os.path.exists(path):
            continue
        prediction_history = pd.read_csv(path)
        rmse_by_hour, mae_by_hour = get_models_hourly_preds_by_window(prediction_history, y_test)
        all_models[folder] = {
            "rmse": rmse_by_hour,
            "mae": mae_by_hour
        }

    fig, ax = plt.subplots(figsize=(10, 4))
    for model_name, metrics in all_models.items():
        ax.plot(range(24), metrics["rmse"], label=MODEL_NAMES[model_name])
    ax.set_xticks(range(24))
    ax.set_xlabel('Prognozės lango valanda',fontsize=STYLE.label_size)
    ax.set_ylabel(STYLE.get_rmse_label_name(),fontsize=STYLE.label_size)
    STYLE.apply(fig, ax)
    output = os.path.join(CURRENT_DIR, 'all_models_rmse_by_window_hours.png')
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    print(output)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    for model_name, metrics in all_models.items():
        ax.plot(range(24), metrics["mae"], label=MODEL_NAMES[model_name])
    ax.set_xticks(range(24))
    ax.set_xlabel('Prognozės lango valanda',fontsize=STYLE.label_size)
    ax.set_ylabel(STYLE.get_mae_label_name(),fontsize=STYLE.label_size)
    STYLE.apply(fig, ax)
    output = os.path.join(CURRENT_DIR, 'all_models_mae_by_window_hours.png')
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    print(output)
        
def get_models_preds_by_hour_of_day(prediction_history, X_test, y_test):
    y_test = np.array(y_test)
    n = len(prediction_history)
    hours = X_test["hour"].values[:n] 

    rmse_by_hour = []
    mae_by_hour = []

    for hour in range(24):
        preds_list = []
        actuals_list = []

        for i in range(n):
            window_start_hour = int(hours[i])
            looked_hour_column = (hour - window_start_hour) % 24
            if i + looked_hour_column < len(y_test):
                preds_list.append(prediction_history.values[i, looked_hour_column])
                actuals_list.append(y_test[i + looked_hour_column])

        rmse_by_hour.append(root_mean_squared_error(actuals_list, preds_list))
        mae_by_hour.append(mean_absolute_error(actuals_list, preds_list))

    return rmse_by_hour, mae_by_hour

def plot_direct_predictions_by_hour_of_day(predictions_folder, X_test, y_test):
    path = os.path.join(CURRENT_DIR, predictions_folder, "direct_predictions.csv")
    prediction_history = pd.read_csv(path)

    rmse_by_hour, mae_by_hour = get_models_preds_by_hour_of_day(prediction_history,X_test,y_test)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(range(24), rmse_by_hour, label=STYLE.get_rmse_label_name())
    ax.plot(range(24), mae_by_hour, label=STYLE.get_mae_label_name())

    ax.set_xticks(range(24))
    ax.set_xlabel("Dienos valanda",fontsize=STYLE.label_size)
    ax.set_ylabel("Paklaida (EUR/MWh)",fontsize=STYLE.label_size)

    STYLE.apply(fig, ax)

    output = os.path.join(CURRENT_DIR, "direct_rmse_mae_by_hour_of_day.png")
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")

    print(output)

def plot_direct_predictions_by_hour_of_day_all(X_test, y_test):
    all_models = {}
    for folder in ["lgbm", "xgb", "mlp", "lstm"]:
        path = os.path.join(CURRENT_DIR, folder, "direct_predictions.csv")
        if not os.path.exists(path):
            continue
        prediction_history = pd.read_csv(path)
        rmse_by_hour, mae_by_hour = get_models_preds_by_hour_of_day(prediction_history, X_test, y_test)
        all_models[folder] = {
            "rmse": rmse_by_hour,
            "mae": mae_by_hour
        }

    fig, ax = plt.subplots(figsize=(10, 4))
    for model_name, metrics in all_models.items():
        ax.plot(range(24), metrics["rmse"], label=MODEL_NAMES[model_name])
    ax.set_xticks(range(24))
    ax.set_xlabel('Dienos valanda',fontsize=STYLE.label_size)
    ax.set_ylabel(STYLE.get_rmse_label_name(),fontsize=STYLE.label_size)
    STYLE.apply(fig, ax)
    output = os.path.join(CURRENT_DIR, 'all_models_rmse_by_hours_of_day.png')
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    print(output)
    
    fig, ax = plt.subplots(figsize=(10, 4))

    for model_name, metrics in all_models.items():
        ax.plot(range(24), metrics["mae"], label=MODEL_NAMES[model_name])
    ax.set_xticks(range(24))
    ax.set_xlabel('Dienos valanda',fontsize=STYLE.label_size)
    ax.set_ylabel(STYLE.get_mae_label_name(),fontsize=STYLE.label_size)
    STYLE.apply(fig, ax)
    output = os.path.join(CURRENT_DIR, 'all_models_mae_by_hours_of_day.png')
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    print(output)
    
def plot_models_hourly_rmae_by_window_hours(y_test):
    baseline_path = os.path.join(CURRENT_DIR, "baseline_predictions.csv")
    baseline_preds = pd.read_csv(baseline_path)

    _, mae_by_hour_baseline = get_models_hourly_preds_by_window(baseline_preds, y_test)

    all_models = {}

    for folder in ["lgbm", "xgb", "mlp", "lstm"]:
        path = os.path.join(CURRENT_DIR, folder, "direct_predictions.csv")
        if not os.path.exists(path):
            continue
        prediction_history = pd.read_csv(path)

        _, mae_by_hour_model = get_models_hourly_preds_by_window(prediction_history, y_test)
        rmae = np.array(mae_by_hour_model) / np.array(mae_by_hour_baseline)
        all_models[folder] = {"mae": mae_by_hour_model, "rmae": rmae}

    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(10, 4))
    for model_name, metrics in all_models.items():
        ax.plot(hours, metrics["rmae"], label=MODEL_NAMES[model_name])
    ax.set_xlabel("Prognozės lango valanda",fontsize=STYLE.label_size)
    ax.set_ylabel(STYLE.get_rmae_label_name(),fontsize=STYLE.label_size)
    ax.set_xticks(range(24))
    STYLE.apply(fig, ax)

    output = os.path.join(CURRENT_DIR, "all_models_rmae_by_window.png")
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    rmae_rows = []
    for model_name, metrics in all_models.items():
        for hour, rmae_val in zip(hours, metrics["rmae"]):
            rmae_rows.append({"model": MODEL_NAMES[model_name], "hour": hour, "rmae": round(rmae_val, 4)})

    rmae_df = pd.DataFrame(rmae_rows)
    csv_path = os.path.join(CURRENT_DIR, "all_models_rmae_by_window.csv")
    rmae_df.pivot(index="hour", columns="model", values="rmae").to_csv(csv_path)
    print(csv_path)
    
def plot_models_hourly_rmae_by_day_hours(X_test,y_test):
    baseline_path = os.path.join(CURRENT_DIR, "baseline_predictions.csv")
    baseline_preds = pd.read_csv(baseline_path)

    _, mae_by_hour_baseline = get_models_preds_by_hour_of_day(baseline_preds,X_test, y_test)

    all_models = {}

    for folder in ["lgbm", "xgb", "mlp", "lstm"]:
        path = os.path.join(CURRENT_DIR, folder, "direct_predictions.csv")
        if not os.path.exists(path):
            continue
        prediction_history = pd.read_csv(path)
        _, mae_by_hour_model = get_models_preds_by_hour_of_day(prediction_history,X_test, y_test)
        rmae = np.array(mae_by_hour_model) / np.array(mae_by_hour_baseline)
        all_models[folder] = {"mae": mae_by_hour_model, "rmae": rmae}

    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(10, 4))
    for model_name, metrics in all_models.items():
        ax.plot(hours, metrics["rmae"], label=MODEL_NAMES[model_name])
    ax.set_xticks(range(24))
    ax.axhline(1.0, linestyle="--", color="black")
    ax.set_xlabel("Dienos valanda",fontsize=STYLE.label_size)
    ax.set_ylabel(STYLE.get_rmae_label_name(),fontsize=STYLE.label_size)

    STYLE.apply(fig, ax)

    output = os.path.join(CURRENT_DIR, "all_models_rmae_by_dayhours.png")
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    print(output)
        
def test_lstm_windows_24(y_test):
    all_models = {}
    for folder in [os.path.join("lstm","3_robust"), 
                   os.path.join("lstm","6_robust"), 
                   os.path.join("lstm","12_robust"),
                   os.path.join("lstm","24_robust"),]:
        path = os.path.join(CURRENT_DIR, folder, "direct_predictions.csv")
        if not os.path.exists(path):
            continue
        prediction_history = pd.read_csv(path)
        rmse_by_hour, mae_by_hour = get_models_hourly_preds_by_window(prediction_history, y_test)
        all_models[folder] = {
            "rmse": rmse_by_hour,
            "mae": mae_by_hour
        }

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ["Su 3 žingsnių įvestimi", "Su 6 žingsnių įvestimi",
              "Su 12 žingsnių įvestimi", "Su 24 žingsnių įvestimi"]
    
    for (model_name, metrics),label in zip(all_models.items(),labels):
        ax.plot(range(24), metrics["mae"], label=label)
    ax.set_xticks(range(24))
    ax.set_xlabel('Prognozės lango valanda',fontsize=STYLE.label_size)
    ax.set_ylabel(STYLE.get_mae_label_name(),fontsize=STYLE.label_size)
    STYLE.apply(fig, ax)
    output = os.path.join(CURRENT_DIR, 'lstm_testing.png')
    fig.savefig(output, dpi=STYLE.dpi, bbox_inches="tight")
    print(output)
    
def test_lstm_inputs(y_test):
    all_maes = {}
    folders = [
        ("lstm/3_robust", "3"),
        ("lstm/6_robust", "6"),
        ("lstm/12_robust", "12"),
        ("lstm/24_robust", "24"),
    ]

    for folder, label in folders:
        path = os.path.join(CURRENT_DIR, folder, "direct_predictions.csv")
        if not os.path.exists(path):
            continue

        prediction_history = pd.read_csv(path)
        _, mae_by_hour = get_models_hourly_preds_by_window(prediction_history,y_test)
        all_maes[label] = mae_by_hour

    mae_df = pd.DataFrame(all_maes)
    result_df = pd.DataFrame({
        "hour": range(24),
        "best_window": mae_df.idxmin(axis=1),
        "best_mae": mae_df.min(axis=1)
    })
    result_df = pd.concat([result_df, mae_df], axis=1)
    averages = mae_df.mean(axis=0)
    averages_df = pd.DataFrame({
        "window": averages.index,
        "average_mae": averages.values
    })
    avg_row = {
        "hour": "AVG",
        "best_window": averages.idxmin(),
        "best_mae": averages.min()
    }
    for col in mae_df.columns:
        avg_row[col] = averages[col]
    result_df = pd.concat([result_df, pd.DataFrame([avg_row])],ignore_index=True)

    output_path = os.path.join(CURRENT_DIR, "lstm_by_hour.csv")
    result_df.to_csv(output_path, index=False)
    averages_output = os.path.join(CURRENT_DIR, "lstm_averages.csv")
    averages_df.to_csv(averages_output, index=False)
    return result_df, averages_df

def tree_statistics(y_test):
    all_maes = {}
    folders = [
        ("lgbm", "LGBM"),
        ("xgb", "XGB")
    ]

    for folder, label in folders:
        path = os.path.join(CURRENT_DIR, folder, "direct_predictions.csv")
        if not os.path.exists(path):
            continue

        prediction_history = pd.read_csv(path)
        _, mae_by_hour = get_models_hourly_preds_by_window(prediction_history,y_test)
        all_maes[label] = mae_by_hour

    mae_df = pd.DataFrame(all_maes)
    result_df = pd.DataFrame({
        "hour": range(24),
        "best_model": mae_df.idxmin(axis=1),
        "best_mae": mae_df.min(axis=1)
    })
    result_df = pd.concat([result_df, mae_df], axis=1)
    averages = mae_df.mean(axis=0)
    averages_df = pd.DataFrame({
        "model": averages.index,
        "average_mae": averages.values
    })
    avg_row = {
        "hour": "AVG",
        "best_model": averages.idxmin(),
        "best_mae": averages.min()
    }
    for col in mae_df.columns:
        avg_row[col] = averages[col]
    result_df = pd.concat([result_df, pd.DataFrame([avg_row])],ignore_index=True)
    output_path = os.path.join(CURRENT_DIR,"trees_by_hour.csv")
    print(output_path)
    result_df.to_csv(output_path, index=False)
    averages_output = os.path.join(CURRENT_DIR,"trees_averages.csv")
    print(averages_output)
    averages_df.to_csv(averages_output, index=False)

    return result_df, averages_df