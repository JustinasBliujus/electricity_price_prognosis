import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.baseline_models import calculate_baseline_models
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from plot_style import PlotStyle

OUTPUT_DIR = "analyze_data"
CSV_PATH   = os.path.join(OUTPUT_DIR, "baseline_results.csv")
N_SPLITS   = 5
TEST_SIZE  = 0.2
EPOCHS     = 100
style = PlotStyle()

def plot_fold_splits(X, y, n_splits, test_size, output_dir):

    split_index = int(len(X) * (1 - test_size))
    X_cv = X[:split_index]
    time_series_split_indexes = TimeSeriesSplit(n_splits=n_splits)

    dates = pd.to_datetime({
        "year":  X["year"],
        "month": X["month"],
        "day":   X["day"],
        "hour":  X["hour"]
    }).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    for fold, (train_index, val_index) in enumerate(time_series_split_indexes.split(X_cv)):
        y_position = fold + 1
        ax.barh(y_position,len(train_index),left=train_index[0],color="steelblue",alpha=1,label="Mokymo duomenys (eil. skaičius)" if fold == 0 else "")
        ax.barh(y_position,len(val_index),left=val_index[0],color="orange",alpha=1,label="Validacijos duomenys (eil. skaičius)" if fold == 0 else "")
        ax.text(train_index[0] + len(train_index) / 2,y_position,str(len(train_index)),va="center",ha="center",fontsize=10,color="white")
        ax.text(val_index[0] + len(val_index) / 2,y_position,str(len(val_index)),va="center",ha="center",fontsize=10,color="black")
        
    test_y = 6
    ax.barh(test_y,split_index,left=0,color="steelblue",alpha=1)
    ax.barh(test_y,len(X) - split_index,left=split_index,color="red",alpha=1,label="Testavimo duomenys (eil. skaičius)")
    ax.text(split_index / 2,test_y,str(split_index),va="center",ha="center",fontsize=10,color="white")
    ax.text(split_index + (len(X) - split_index) / 2,test_y,str(len(X) - split_index),va="center",ha="center",fontsize=10,color="white")
    ax.set_yticks(list(range(1, n_splits + 2)))
    ax.set_yticklabels([f"{i+1} validacijos etapas" for i in range(n_splits)] + ["Testavimo etapas"])

    n_ticks = 5
    index = np.linspace(0, len(dates) - 1, n_ticks)
    index = np.round(index).astype(int)

    ax.set_xticks(index)
    ax.set_xticklabels(dates.iloc[index].dt.strftime("%Y-%m-%d"))
    ax.set_xlabel("Data", fontsize=style.label_size)
    ax.set_ylabel("Validacijos ir testavimo etapai", fontsize=style.label_size)
    style.apply(fig, ax)

    path = os.path.join(output_dir, "fold_splits.png")
    plt.savefig(path, dpi=style.dpi, bbox_inches="tight")
    print(path)
    
def get_baselines(X, y):
    plot_fold_splits(X, y, N_SPLITS, TEST_SIZE, OUTPUT_DIR)

    X = X.to_numpy()
    y = y.to_numpy()

    baseline_results = calculate_baseline_models(X, y)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        {"date": now, "split": "test", "model": "lag1", "rmse": baseline_results["test"]["lag1"][0], "mae": baseline_results["test"]["lag1"][1]},
        {"date": now, "split": "test", "model": "mean", "rmse": baseline_results["test"]["mean"][0], "mae": baseline_results["test"]["mean"][1]},
        {"date": now, "split": "test", "model": "rolling_mean_24", "rmse": baseline_results["test"]["rolling_mean_24"][0], "mae": baseline_results["test"]["rolling_mean_24"][1]},
    ]

    for fold in baseline_results["cv"]:
        rows.append({"date": now, "split": f"cv_{fold['fold']}", "model": "lag1", "rmse": fold["lag1_rmse"], "mae": fold["lag1_mae"]})
        rows.append({"date": now, "split": f"cv_{fold['fold']}", "model": "mean", "rmse": fold["mean_rmse"], "mae": fold["mean_mae"]})
        rows.append({"date": now, "split": f"cv_{fold['fold']}", "model": "rolling_mean_24", "rmse": fold["rolling_mean_24_rmse"], "mae": fold["rolling_mean_24_mae"]})

    if os.path.exists(CSV_PATH):
        df = pd.concat([pd.read_csv(CSV_PATH), pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)

    df.to_csv(CSV_PATH, index=False)

    print(CSV_PATH)
