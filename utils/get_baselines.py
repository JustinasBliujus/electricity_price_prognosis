import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.baseline_models import calculate_baseline_models
from lgbm.lgbm import lgbm_run
from xgb.xgboost import xgb_run
from mlp.mlp import mlp_run
from lstm.lstm import lstm_run
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

OUTPUT_DIR = "analyze_data"
CSV_PATH   = os.path.join(OUTPUT_DIR, "baseline_results.csv")
N_SPLITS   = 5
TEST_SIZE  = 0.2
EPOCHS     = 100

def plot_fold_splits(X, y, n_splits, test_size, output_dir):
    split_idx = int(len(X) * (1 - test_size))
    X_cv      = X[:split_idx]
    
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    fig, ax = plt.subplots(figsize=(14, 6))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        ax.barh(fold + 1, len(train_idx), left=train_idx[0], color='steelblue', alpha=0.6, label='Train' if fold == 0 else "")
        ax.barh(fold + 1, len(val_idx),   left=val_idx[0],   color='orange',    alpha=0.8, label='Val'   if fold == 0 else "")
        ax.text(train_idx[0] + len(train_idx) / 2, fold + 1, f'{len(train_idx)}', va='center', ha='center', fontsize=8, color='white')
        ax.text(val_idx[0]   + len(val_idx)   / 2, fold + 1, f'{len(val_idx)}',   va='center', ha='center', fontsize=8, color='black')

    ax.barh(0, len(X) - split_idx, left=split_idx, color='red', alpha=0.6, label='Test')
    ax.text(split_idx + (len(X) - split_idx) / 2, 0, f'{len(X) - split_idx}', va='center', ha='center', fontsize=8, color='white')

    ax.set_yticks(range(0, n_splits + 1))
    ax.set_yticklabels(['Test'] + [f'Fold {i+1}' for i in range(n_splits)])
    ax.set_xlabel("Sample index")
    ax.set_title("Time Series Split — Train / Validation / Test")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fold_splits.png"))
    plt.close()

def plot_cv_results(model_name, results, output_dir):
    cv_folds = [r for r in results if r["type"] == "cv"]
    folds    = [r["fold"] for r in cv_folds]
    rmses    = [r["rmse"] for r in cv_folds]
    test_rmse = next(r["rmse"] for r in results if r["type"] == "test")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(folds, rmses, marker='o', label='CV Fold RMSE', color='steelblue')
    ax.axhline(y=test_rmse, color='red', linestyle='--', label=f'Test RMSE: {test_rmse:.4f}')
    cv_mean = np.mean(rmses)
    ax.axhline(y=cv_mean, color='orange', linestyle='--', label=f'CV mean RMSE: {cv_mean:.4f}')
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{model_name} — CV vs Test RMSE")
    ax.legend()
    ax.set_xticks(folds)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_cv_results.png"))
    plt.close()

def plot_model_comparison(model_groups, output_dir):
    fig, axes = plt.subplots(1, len(model_groups), figsize=(14, 5))

    for ax, (group_name, models) in zip(axes, model_groups.items()):
        for model_name, results in models.items():
            cv_folds  = [r for r in results if r["type"] == "cv"]
            folds     = [r["fold"] for r in cv_folds]
            rmses     = [r["rmse"] for r in cv_folds]
            ax.plot(folds, rmses, marker='o', label=model_name)

        ax.set_title(group_name)
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.set_xticks(folds)
        ax.grid(True, alpha=0.3)

    plt.suptitle("CV Fold RMSE Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_cv.png"))
    plt.close()

def plot_fold_detail(X, y, n_splits, test_size, fold_n, output_dir, timestamps=None):
    split_idx = int(len(X) * (1 - test_size))
    X_cv, y_cv = X[:split_idx], y[:split_idx]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        if fold + 1 == fold_n:
            y_train = y_cv[train_idx]
            y_val   = y_cv[val_idx]

            fig, axes = plt.subplots(3, 1, figsize=(14, 10))

            ax = axes[0]
            ax.plot(train_idx, y_train, color='steelblue', label=f'Train ({len(train_idx)} samples)')
            ax.plot(val_idx,   y_val,   color='orange',    label=f'Val ({len(val_idx)} samples)')
            ax.axvline(x=val_idx[0], color='red', linestyle='--', label='Train/Val split')
            ax.set_title(f"Fold {fold_n} — Target values over time")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.hist(y_train, bins=50, alpha=0.6, color='steelblue', label='Train', density=True)
            ax.hist(y_val,   bins=50, alpha=0.6, color='orange',    label='Val',   density=True)
            ax.set_title(f"Fold {fold_n} — Target distribution: Train vs Val")
            ax.set_xlabel("Price")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[2]
            ax.axis('off')
            stats = {
                "":        ["Min", "Max", "Mean", "Std", "Median"],
                "Train":   [f"{y_train.min():.2f}", f"{y_train.max():.2f}", f"{y_train.mean():.2f}", f"{y_train.std():.2f}", f"{np.median(y_train):.2f}"],
                "Val":     [f"{y_val.min():.2f}",   f"{y_val.max():.2f}",   f"{y_val.mean():.2f}",   f"{y_val.std():.2f}",   f"{np.median(y_val):.2f}"],
            }
            table = ax.table(
                cellText  = [stats["Train"], stats["Val"]],
                rowLabels = ["Train", "Val"],
                colLabels = stats[""],
                cellLoc   = 'center',
                loc       = 'center'
            )
            table.scale(1, 2)
            ax.text(0.5, 0.95, f"Fold {fold_n} — Statistics",transform=ax.transAxes, ha='center', va='top', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"fold_{fold_n}_detail.png"))
            plt.close()
            break

def get_baselines(X, y, y_train, y_test, X_train, X_test):
    plot_fold_splits(X, y, N_SPLITS, TEST_SIZE, OUTPUT_DIR)
    X = X.to_numpy()
    y = y.to_numpy()
    plot_fold_detail(X, y, N_SPLITS, TEST_SIZE, fold_n=2, output_dir=OUTPUT_DIR)
    plot_fold_detail(X, y, N_SPLITS, TEST_SIZE, fold_n=3, output_dir=OUTPUT_DIR)

    baseline_results        = calculate_baseline_models(y_train, y_test, X_test)
    lgbm_results_regression = lgbm_run(X, y, n_splits=N_SPLITS, test_size=TEST_SIZE, objective='regression')
    lgbm_results_huber      = lgbm_run(X, y, n_splits=N_SPLITS, test_size=TEST_SIZE, objective='huber')
    lgbm_results_fair       = lgbm_run(X, y, n_splits=N_SPLITS, test_size=TEST_SIZE, objective='fair')
    xgb_results_regression  = xgb_run(X, y, n_splits=N_SPLITS, test_size=TEST_SIZE, objective='reg:squarederror')
    xgb_results_huber       = xgb_run(X, y, n_splits=N_SPLITS, test_size=TEST_SIZE, objective='reg:pseudohubererror')
    mlp_results             = mlp_run(X, y, n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS)
    lstm_results            = lstm_run(X, y, n_splits=N_SPLITS, test_size=TEST_SIZE, epochs=EPOCHS)

    plot_cv_results("lgbm_regression", lgbm_results_regression["results"], OUTPUT_DIR)
    plot_cv_results("lgbm_huber",      lgbm_results_huber["results"],      OUTPUT_DIR)
    plot_cv_results("lgbm_fair",       lgbm_results_fair["results"],       OUTPUT_DIR)
    plot_cv_results("xgb_regression",  xgb_results_regression["results"],  OUTPUT_DIR)
    plot_cv_results("xgb_huber",       xgb_results_huber["results"],       OUTPUT_DIR)
    plot_cv_results("mlp",             mlp_results["results"],             OUTPUT_DIR)
    plot_cv_results("lstm",            lstm_results["results"],            OUTPUT_DIR)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        {"date": now, "model": "last_value",      "rmse": baseline_results["last_value"][1]},
        {"date": now, "model": "train_mean",       "rmse": baseline_results["train_mean"][1]},
        {"date": now, "model": "rolling_mean",     "rmse": baseline_results["rolling_mean"][1]},
        {"date": now, "model": "lgbm_regression",  "rmse": lgbm_results_regression["test_rmse"], "mean_cv_rmse": lgbm_results_regression["mean_cv_rmse"]},
        {"date": now, "model": "lgbm_huber",       "rmse": lgbm_results_huber["test_rmse"],       "mean_cv_rmse": lgbm_results_huber["mean_cv_rmse"]},
        {"date": now, "model": "lgbm_fair",        "rmse": lgbm_results_fair["test_rmse"],        "mean_cv_rmse": lgbm_results_fair["mean_cv_rmse"]},
        {"date": now, "model": "xgb_regression",   "rmse": xgb_results_regression["test_rmse"],   "mean_cv_rmse": xgb_results_regression["mean_cv_rmse"]},
        {"date": now, "model": "xgb_huber",        "rmse": xgb_results_huber["test_rmse"],        "mean_cv_rmse": xgb_results_huber["mean_cv_rmse"]},
        {"date": now, "model": "mlp",              "rmse": mlp_results["test_rmse"],              "mean_cv_rmse": mlp_results["mean_cv_rmse"]},
        {"date": now, "model": "lstm",             "rmse": lstm_results["test_rmse"],             "mean_cv_rmse": lstm_results["mean_cv_rmse"]},
    ]

    model_results = {
        "lgbm_regression": lgbm_results_regression["results"],
        "lgbm_huber":      lgbm_results_huber["results"],
        "lgbm_fair":       lgbm_results_fair["results"],
        "xgb_regression":  xgb_results_regression["results"],
        "xgb_huber":       xgb_results_huber["results"],
        "mlp":             mlp_results["results"],
        "lstm":            lstm_results["results"],
    }

    fold_rows = []
    for model_name, results in model_results.items():
        for r in results:
            fold_rows.append({"date": now, "model": model_name, **r})

    fold_csv_path = os.path.join(OUTPUT_DIR, "baseline_fold_results.csv")
    if os.path.exists(fold_csv_path):
        df_folds = pd.concat([pd.read_csv(fold_csv_path), pd.DataFrame(fold_rows)], ignore_index=True)
    else:
        df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(fold_csv_path, index=False)

    if os.path.exists(CSV_PATH):
        df = pd.concat([pd.read_csv(CSV_PATH), pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    plot_model_comparison({
        "LGBM objectives": {
            "regression": lgbm_results_regression["results"],
            "huber":      lgbm_results_huber["results"],
            "fair":       lgbm_results_fair["results"],
        },
        "XGB objectives": {
            "squarederror":      xgb_results_regression["results"],
            "pseudohubererror":  xgb_results_huber["results"],
        }
    }, OUTPUT_DIR)
    
    print(f"\nResults saved to {CSV_PATH}")
    print(f"Fold results saved to {fold_csv_path}")