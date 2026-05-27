import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import PlotStyle
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

BASE_DIR = base_dir = os.path.dirname(__file__)

def split(X):
        split_index = int(len(X) * 0.8)
        return X[:split_index], X[split_index:]
    
def plot_models_side_by_side(X, start=2000, end=2500):
    style = PlotStyle()
    X_cv, X_test= split(X)
    time_series_split_indexes = TimeSeriesSplit(n_splits=5)
    fold_indexes= list(time_series_split_indexes.split(X_cv))
    _, val_indexes = fold_indexes[1]
    X_val = X_cv.iloc[val_indexes].reset_index(drop=True)
    dates = pd.to_datetime({
        "year":  X_val["year"],
        "month": X_val["month"],
        "day":   X_val["day"],
        "hour":  X_val["hour"]
    })
    csv_files = sorted([
        csv for csv in os.listdir(BASE_DIR)
        if csv.endswith(".csv")
    ])
    models = {
        "lgbm": "LightGBM",
        "xgb":  "XGBoost",
        "mlp":  "Daugiasluoksnis perceptronas",
        "lstm": "Ilgos trumpalaikės atminties tinklas",
    }

    n = len(csv_files)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, file_name in zip(axes, csv_files):
        df = pd.read_csv(os.path.join(BASE_DIR, file_name))
        actual    = df["actual"].values[start:end]
        predicted = df["predicted"].values[start:end]
        x_range = dates.iloc[start:end]

        model_key  = file_name.split("_")[0].lower()
        model_name = models.get(model_key, model_key.upper())

        ax.plot(x_range, actual, label="Faktinės kainos")
        ax.plot(x_range, predicted, label="Prognozė")
        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Data", fontsize=style.label_size)
        ax.set_ylabel("Kaina (EUR/MWh)", fontsize=style.label_size)
        idx = np.linspace(0, len(x_range) - 1, 5, dtype=int)
        ax.set_xticks(x_range.iloc[idx])
        ax.set_xticklabels(x_range.iloc[idx].dt.strftime("%Y-%m-%d"))
        style.apply(fig,ax)
    path = os.path.join(BASE_DIR, "comparison_spike.png")
    plt.savefig(path, dpi=style.dpi, bbox_inches="tight")
    print(path)