import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
from plot_style import PlotStyle

BASE_DIR = base_dir = os.path.dirname(__file__)

style = PlotStyle()

model_map = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "mlp": "Daugiasluoksnis perceptronas",
    "lstm": "Ilgos trumpalaikės atminties tinklas"
}

WEEKDAY_NAMES = ["Pirmadienis", "Antradienis", "Trečiadienis",
                 "Ketvirtadienis", "Penktadienis", "Šeštadienis", "Sekmadienis"]

def plot_averaged():
    plot_all_rmses()
    plot_all_maes()
    plot_all_rmaes()    
    
def plot_all_maes():
    csv_path = os.path.join(BASE_DIR, "all_models_rmae.csv")
    df = pd.read_csv(csv_path)
    df["model_name"] = df["model"].map(model_map)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in df.groupby("model_name"):
        ax.plot(group["fold"], group["avg_mae"], marker="o", label=model)
    ax.set_xlabel("Etapas", fontsize=style.label_size)
    ax.set_ylabel(style.get_mae_label_name(), fontsize=style.label_size)
    ax.legend()
    
    order_labels = [
        "1 Validacija",
        "2 Validacija",
        "3 Validacija",
        "4 Validacija",
        "5 Validacija",
        "Testavimas"
    ]

    ax.set_xticks(range(len(order_labels)))
    ax.set_xticklabels(order_labels)
    style.apply(fig,ax)
    path = os.path.join(BASE_DIR, "avg_maes.png")
    plt.savefig(path, dpi = style.dpi, bbox_inches="tight")
    print(path)

def plot_all_rmaes():
    csv_path = os.path.join(BASE_DIR, "all_models_rmae.csv")
    df = pd.read_csv(csv_path)
    df["model_name"] = df["model"].map(model_map)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in df.groupby("model_name"):
        ax.plot(group["fold"], group["rmae"], marker="o", label=model)
    ax.set_xlabel("Etapas", fontsize=style.label_size)
    ax.set_ylabel(style.get_rmae_label_name(), fontsize=style.label_size)
    ax.legend()
    
    order_labels = [
        "1 Validacija",
        "2 Validacija",
        "3 Validacija",
        "4 Validacija",
        "5 Validacija",
        "Testavimas"
    ]

    ax.set_xticks(range(len(order_labels)))
    ax.set_xticklabels(order_labels)
    style.apply(fig,ax)
    path = os.path.join(BASE_DIR, "avg_rmae.png")
    plt.savefig(path, dpi = style.dpi,bbox_inches="tight")
    print(path)
    
def plot_all_rmses():
    csv_path = os.path.join(BASE_DIR, "all_models_rmse.csv")
    df = pd.read_csv(csv_path)
    df["model_name"] = df["model"].map(model_map)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in df.groupby("model_name"):
        ax.plot(group["fold"], group["avg_rmse"], marker="o", label=model)
    ax.set_xlabel("Etapas", fontsize=style.label_size)
    ax.set_ylabel(style.get_rmse_label_name(), fontsize=style.label_size)
    ax.legend()
    
    order_labels = [
        "1 Validacija",
        "2 Validacija",
        "3 Validacija",
        "4 Validacija",
        "5 Validacija",
        "Testavimas"
    ]
    ax.set_xticks(range(len(order_labels)))
    ax.set_xticklabels(order_labels)
    style.apply(fig,ax)
    path = os.path.join(BASE_DIR, "avg_rmse.png")
    plt.savefig(path, dpi = style.dpi,bbox_inches="tight")
    print(path)
    
def make_df(X, folders_name):
    base_dir = os.path.join(BASE_DIR, folders_name)
    all_dfs = []

    for folder in sorted(os.listdir(base_dir)):
        if not folder.startswith(folders_name):
            continue
        predictions_path = os.path.join(base_dir, folder, "results", "predictions.csv")
        if os.path.exists(predictions_path):
            all_dfs.append(pd.read_csv(predictions_path))

    if not all_dfs:
        print("no predictions found")
        return None, None

    df = pd.concat(all_dfs).groupby(level=0).mean().reset_index(drop=True)

    X_test = X[int(len(X) * 0.8):].reset_index(drop=True)
    X_test = X_test.iloc[len(X_test) - len(df):].reset_index(drop=True)

    df["weekday"] = X_test["dayofweek"].values
    df["hour"]    = X_test["hour"].values
    df["date"]    = pd.to_datetime({
        "year":  X_test["year"],
        "month": X_test["month"],
        "day":   X_test["day"],
        "hour":  X_test["hour"],
    })
    return df, X_test

def plot_hourly_test_results(X, folders_name):
    df, _ = make_df(X,folders_name)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(df["date"], df["actual"], label="Faktinė kaina")
    plt.plot(df["date"], df["predicted"], label="Prognozuota kaina")
    plt.xlabel("Data", fontsize=style.label_size)
    plt.ylabel("Kaina (EUR/MWh)", fontsize=style.label_size)
    
    style.apply(fig,ax)
    path = os.path.join(base_dir, "hourly_test.png")
    plt.savefig(path, dpi = style.dpi,bbox_inches="tight")
    print(path)

    return df

def plot_mae_by_feature(X, folders_name):
    base_dir = os.path.join(BASE_DIR, folders_name)
    df, X = make_df(X, folders_name)

    df["abs_error"] = df["error"].abs()
    df["hour"]    = X["hour"].iloc[:len(df)].values
    df["weekday"] = X["dayofweek"].iloc[:len(df)].values

    mae_hour = df.groupby("hour")["abs_error"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.bar(mae_hour.index, mae_hour.values,alpha=1)
    plt.xlabel("Valanda", fontsize=style.label_size)
    plt.ylabel(style.get_mae_label_name(), fontsize=style.label_size)
    plt.xticks(range(24))
    style.apply(fig,ax)
    path = os.path.join(base_dir, "mae_hourly.png")
    plt.savefig(path, dpi = style.dpi,bbox_inches="tight")
    print(path)
    
    mae_weekday = df.groupby("weekday")["abs_error"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.bar([WEEKDAY_NAMES[i] for i in mae_weekday.index], mae_weekday.values,alpha=1)
    plt.xlabel("Savaitės diena", fontsize=style.label_size)
    plt.ylabel(style.get_mae_label_name(), fontsize=style.label_size)
    style.apply(fig,ax)
    path = os.path.join(base_dir, "mae_weekdays.png")
    plt.savefig(path, dpi = style.dpi,bbox_inches="tight")
    print(path)

def plot_all_models_mae_by_features(X, folders_names=["lgbm", "xgb", "mlp", "lstm"]):
    results = {}
    for model in folders_names:
        df, _ = make_df(X, model)
        df["abs_error"] = df["error"].abs()
        results[model] = df

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, model in zip(axes.flatten(), folders_names):
        df = results[model]
        mae_hour = df.groupby("hour")["abs_error"].mean()
        ax.bar(mae_hour.index, mae_hour.values, alpha=1)
        ax.set_title(model_map.get(model, model))
        ax.set_xlabel("Valanda",fontsize=style.label_size)
        ax.set_ylabel(style.get_mae_label_name(),fontsize=style.label_size)
        ax.set_xticks(range(24))
        style.apply(fig, ax)

    path = os.path.join(BASE_DIR, "all_mae_hour.png")
    fig.savefig(path, dpi=style.dpi, bbox_inches="tight")
    print(path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax, model in zip(axes.flatten(), folders_names):
        df = results[model]
        mae_weekday = df.groupby("weekday")["abs_error"].mean()
        ax.bar(
            [WEEKDAY_NAMES[i] for i in mae_weekday.index],
            mae_weekday.values,
            alpha=1
        )
        ax.set_xlabel("Savaitės diena",fontsize=style.label_size)
        ax.set_ylabel(style.get_mae_label_name(),fontsize=style.label_size)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(model_map.get(model, model))
        style.apply(fig, ax)
        
    path = os.path.join(BASE_DIR, "all_mae_weekday.png")
    fig.savefig(path, dpi=style.dpi, bbox_inches="tight")
    print(path)