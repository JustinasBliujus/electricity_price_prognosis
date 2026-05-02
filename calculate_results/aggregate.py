import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

BASELINE_MAE = [19.66081055607917,15.446255105246623,15.13860508953817,25.087891925856113,20.808834432924915,20.287543969849246]
FOLD_ORDER = ["1", "2", "3", "4", "5", "test"]
MODEL_FOLDERS = ["lgbm", "xgb", "mlp", "lstm"]

def plot_baseline_fold2(X, y, n_splits, test_size):
    split_idx = int(len(X) * (1 - test_size))
    X_cv = X[:split_idx]
    y_cv = y[:split_idx]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_indices = list(tscv.split(X_cv))
    train_idx, val_idx = fold_indices[1]

    y_val   = y_cv.iloc[val_idx]
    lag1_val = X_cv.iloc[val_idx]["lag_1"]

    plt.figure(figsize=(12, 5))
    plt.plot(y_val.values,    label="Realios reikšmės", linewidth=1)
    plt.plot(lag1_val.values, label="Atsilikimas 1h",    linewidth=1, linestyle="--")
    plt.xlabel("Imtys")
    plt.ylabel("Reikšmė")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "fold2_baseline.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to: {os.path.abspath(save_path)}")

def aggregate(folders_name):
    base_dir = os.path.join(os.path.dirname(__file__), folders_name)
    rows = []

    for folder in os.listdir(base_dir):
        if not folder.startswith(folders_name):
            continue

        metrics_path = os.path.join(base_dir, folder, "results", "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        df = pd.read_csv(metrics_path)
        df["model"] = folder
        rows.append(df)

    all_data = pd.concat(rows, ignore_index=True)

    summary = all_data.groupby("fold")["mae"].mean().reset_index()
    summary.columns = ["fold", "avg_mae"]

    cv_rows   = summary[summary["fold"] != "test"]
    test_rows = summary[summary["fold"] == "test"]
    summary   = pd.concat([cv_rows, test_rows], ignore_index=True)

    print(summary)
    summary.to_csv(os.path.join(base_dir, folders_name+"_avg_mae.csv"), index=False)

def compare_relative_mae():
    base_dir = os.path.dirname(__file__)
    baseline = dict(zip(FOLD_ORDER, BASELINE_MAE))

    all_rows = []

    for model in MODEL_FOLDERS:
        model_dir = os.path.join(base_dir, model)
        if not os.path.exists(model_dir):
            continue

        csv_path = os.path.join(model_dir, f"{model}_avg_mae.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df["fold"] = df["fold"].astype(str)
        df["baseline_mae"] = df["fold"].map(baseline)
        df["rmae"] = df["avg_mae"] / df["baseline_mae"]
        df["model"] = model
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result = result[["model", "fold", "avg_mae", "baseline_mae", "rmae"]]

    print(result)
    result.to_csv(os.path.join(base_dir, "all_models_rmae.csv"), index=False)

MODEL_NAMES = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "mlp": "Daugiasluoksnis perceptronas",
    "lstm": "Ilgos trumpalaikės atminties tinklas"
}

def plot_from_csv():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "all_models_rmae.csv")

    df = pd.read_csv(csv_path)
    df["fold"] = df["fold"].astype(str)

    fold_order = ["1", "2", "3", "4", "5", "test"]
    df["fold"] = pd.Categorical(df["fold"], categories=fold_order, ordered=True)
    df = df.sort_values(["model", "fold"])

    data_dict = {}
    for model, group in df.groupby("model"):
        data_dict[model] = group["rmae"].tolist()

    plot_metrics(data_dict, ylabel="Reliatyvi absoliuti paklaida",
             save_path=os.path.join(base_dir, "rmae_plot.png"))

def plot_metrics(data_dict, ylabel, save_path=None):
    labels = [
        "Validacija 1",
        "Validacija 2",
        "Validacija 3",
        "Validacija 4",
        "Validacija 5",
        "Testavimas"
    ]

    plt.figure(figsize=(10, 6))

    for model_name, values in data_dict.items():
        if values:
            label = MODEL_NAMES.get(model_name, model_name)
            plt.plot(labels, values, marker='o', label=label)

    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        print(f"Saved to: {os.path.abspath(save_path)}")
        plt.savefig(save_path, dpi=150)

def aggregate_rmse(folders_name):
    base_dir = os.path.join(os.path.dirname(__file__), folders_name)
    rows = []

    for folder in os.listdir(base_dir):
        if not folder.startswith(folders_name):
            continue

        metrics_path = os.path.join(base_dir, folder, "results", "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        df = pd.read_csv(metrics_path)
        df["model"] = folder
        rows.append(df)

    all_data = pd.concat(rows, ignore_index=True)

    summary = all_data.groupby("fold")["rmse"].mean().reset_index()
    summary.columns = ["fold", "avg_rmse"]
    cv_rows   = summary[summary["fold"].astype(str) != "test"]
    test_rows = summary[summary["fold"].astype(str) == "test"]
    summary   = pd.concat([cv_rows, test_rows], ignore_index=True)

    print(summary)
    out_path = os.path.join(base_dir, f"{folders_name}_avg_rmse.csv")
    summary.to_csv(out_path, index=False)
    return out_path


def collect_all_rmse():
    base_dir = os.path.dirname(__file__)
    all_rows = []

    for model in MODEL_FOLDERS:
        model_dir = os.path.join(base_dir, model)
        csv_path  = os.path.join(model_dir, f"{model}_avg_rmse.csv")

        if not os.path.exists(csv_path):
  
            aggregate_rmse(model)

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        df["fold"]  = df["fold"].astype(str)
        df["model"] = model
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result = result[["model", "fold", "avg_rmse"]]

    out_path = os.path.join(base_dir, "all_models_rmse.csv")
    result.to_csv(out_path, index=False)
    print(result)
    return result


def plot_rmse():
    df = collect_all_rmse()

    fold_order = ["1", "2", "3", "4", "5", "test"]
    df["fold"] = pd.Categorical(df["fold"], categories=fold_order, ordered=True)
    df = df.sort_values(["model", "fold"])

    data_dict = {}
    for model, group in df.groupby("model"):
        data_dict[model] = group["avg_rmse"].tolist()

    base_dir = os.path.dirname(__file__)
    plot_metrics(
        data_dict,
        ylabel="Vidutinė kvadratinė paklaidos šaknis",
        save_path=os.path.join(base_dir, "rmse_plot.png")
    )

def collect_all_mae():
    base_dir = os.path.dirname(__file__)
    all_rows = []

    for model in MODEL_FOLDERS:
        model_dir = os.path.join(base_dir, model)
        csv_path  = os.path.join(model_dir, f"{model}_avg_mae.csv")

        if not os.path.exists(csv_path):
            aggregate(model) 

        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        df["fold"]  = df["fold"].astype(str)
        df["model"] = model
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result = result[["model", "fold", "avg_mae"]]
    print(result)
    return result


def plot_mae():
    df = collect_all_mae()

    fold_order = ["1", "2", "3", "4", "5", "test"]
    df["fold"] = pd.Categorical(df["fold"], categories=fold_order, ordered=True)
    df = df.sort_values(["model", "fold"])

    data_dict = {}
    for model, group in df.groupby("model"):
        data_dict[model] = group["avg_mae"].tolist()

    base_dir = os.path.dirname(__file__)
    plot_metrics(
        data_dict,
        ylabel="Vidutinė absoliuti paklaida",
        save_path=os.path.join(base_dir, "mae_plot.png")
    )