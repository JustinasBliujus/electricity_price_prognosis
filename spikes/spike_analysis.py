import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_models_side_by_side(folder, start=2000, end=2500):

    csv_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])

    if not csv_files:
        print("No CSV files found in folder.")
        return

    model_labels = {
        "lgbm": "LightGBM",
        "xgb":  "XGBoost",
        "mlp":  "Daugiasluoksnis perceptronas",
        "lstm": "Ilgos trumpalaikės atminties tinklas",
    }

    n = len(csv_files)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, fname in zip(axes, csv_files):
        df = pd.read_csv(os.path.join(folder, fname))
        actual    = df["actual"].values[start:end]
        predicted = df["predicted"].values[start:end]
        x_range   = range(start, start + len(actual))

        model_key  = fname.split("_")[0].lower()
        model_name = model_labels.get(model_key, model_key.upper())

        ax.plot(x_range, actual,    linewidth=1,   label="Realios reikšmės")
        ax.plot(x_range, predicted, linewidth=1.2, linestyle="--", label=f"{model_name} prognozė")
        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Imtys")
        ax.set_ylabel("Reikšmė")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(folder, "comparison_2000_2500.png")
    plt.savefig(out_path, dpi=150)
    print(f"Comparison plot saved to {out_path}")
    plt.close()