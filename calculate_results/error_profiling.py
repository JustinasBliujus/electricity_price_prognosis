import os
import pandas as pd
import matplotlib.pyplot as plt

def get_hourly_test_results(X, folders_name):
    base_dir = os.path.join(os.path.dirname(__file__), folders_name)

    first_folder = None
    for folder in sorted(os.listdir(base_dir)):
        if folder.startswith(folders_name):
            predictions_path = os.path.join(base_dir, folder, "results", "predictions.csv")
            if os.path.exists(predictions_path):
                first_folder = predictions_path
                break

    if first_folder is None:
        print("No predictions found.")
        return None, None

    X_test = X[int(len(X) * 0.8):].reset_index(drop=True)
    df = pd.read_csv(first_folder).reset_index(drop=True)

    X_test = X_test.iloc[len(X_test) - len(df):].reset_index(drop=True)

    df["weekday"] = X_test["dayofweek"].values
    df["hour"]    = X_test["hour"].values
    df["date"]    = pd.to_datetime({
        "year":  X_test["year"],
        "month": X_test["month"],
        "day":   X_test["day"],
        "hour":  X_test["hour"],
    })

    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["actual"], label="Faktinė", linewidth=1)
    plt.plot(df["date"], df["predicted"], label="Prognozuota", linewidth=1)
    plt.xlabel("Data")
    plt.ylabel("Kaina (EUR/MWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "hourly_test.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return df, X_test


def plot_mae_by_feature(X, folders_name):
    base_dir = os.path.join(os.path.dirname(__file__), folders_name)
    df, X = get_hourly_test_results(X, folders_name)

    df["abs_error"] = df["error"].abs()
    df["hour"]    = X["hour"].iloc[:len(df)].values
    df["weekday"] = X["dayofweek"].iloc[:len(df)].values

    WEEKDAY_NAMES = ["Pirmadienis", "Antradienis", "Trečiadienis",
                     "Ketvirtadienis", "Penktadienis", "Šeštadienis", "Sekmadienis"]

    mae_hour = df.groupby("hour")["abs_error"].mean()

    plt.figure(figsize=(10, 4))
    plt.bar(mae_hour.index, mae_hour.values)
    plt.xlabel("Valanda")
    plt.ylabel("Absoliuti vidutinė paklaida")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "mae_hourly.png"))

    mae_weekday = df.groupby("weekday")["abs_error"].mean()

    plt.figure(figsize=(10, 4))
    plt.bar([WEEKDAY_NAMES[i] for i in mae_weekday.index], mae_weekday.values)
    plt.xlabel("Savaitės diena")
    plt.ylabel("MAE")
    plt.title(f"{folders_name} — MAE pagal savaitės dieną")
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Weekdays.png"))

MODEL_NAMES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "mlp": "Daugiasluoksnis perceptronas",
    "lstm": "Ilgos trumpalaikės atminties tinklas"
}

WEEKDAY_NAMES = ["Pirmadienis", "Antradienis", "Trečiadienis",
                 "Ketvirtadienis", "Penktadienis", "Šeštadienis", "Sekmadienis"]

def plot_all_models(X, folders_names=["lgbm", "xgb", "mlp", "lstm"]):
    base_dir = os.path.dirname(__file__)

    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=False)
    for ax, model in zip(axes, folders_names):
        df, _ = get_hourly_test_results(X, model)
        if df is None:
            continue
        ax.plot(df["date"], df["actual"], label="Faktinė", linewidth=1)
        ax.plot(df["date"], df["predicted"], label="Prognozuota", linewidth=1)
        ax.set_title(MODEL_NAMES.get(model, model))
        ax.set_ylabel("Kaina (EUR/MWh)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "all_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for ax, model in zip(axes.flatten(), folders_names):
        df, _ = get_hourly_test_results(X, model)
        if df is None:
            continue
        df["abs_error"] = df["error"].abs()
        mae_hour = df.groupby("hour")["abs_error"].mean()
        ax.bar(mae_hour.index, mae_hour.values)
        ax.set_title(MODEL_NAMES.get(model, model))
        ax.set_xlabel("Valanda")
        ax.set_ylabel("Absoliuti vidutinė paklaida")
        ax.set_xticks(range(24))
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "all_mae_hour.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, model in zip(axes.flatten(), folders_names):
        df, _ = get_hourly_test_results(X, model)
        if df is None:
            continue
        df["abs_error"] = df["error"].abs()
        mae_weekday = df.groupby("weekday")["abs_error"].mean()
        ax.bar([WEEKDAY_NAMES[i] for i in mae_weekday.index], mae_weekday.values)
        ax.set_title(MODEL_NAMES.get(model, model))
        ax.set_xlabel("Savaitės diena")
        ax.set_ylabel("Absoliuti vidutinė paklaida")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
        for ax in axes.flatten():
            ax.tick_params(axis="y", labelleft=True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "all_mae_weekday.png"), dpi=150, bbox_inches="tight")
    plt.close()