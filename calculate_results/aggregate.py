import os
import pandas as pd

BASELINE_MAE = [19.66081055607917,15.446255105246623,15.13860508953817,25.087891925856113,20.808834432924915,20.287543969849246]
FOLD_ORDER = ["1", "2", "3", "4", "5", "test"]
MODEL_FOLDERS = ["lgbm", "xgb", "mlp", "lstm"]
BASE_DIR = base_dir = os.path.dirname(__file__)

def collect_rmse_and_rmae():
    for folder in MODEL_FOLDERS:
        aggregate_avg_mae_per_fold(folder)
        aggregate_avg_rmse_per_fold(folder)
    compare_relative_mae()
    collect_all_rmse()

def aggregate_avg_mae_per_fold(folders_name):
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

def aggregate_avg_rmse_per_fold(folders_name):
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
            aggregate_avg_rmse_per_fold(model)

        if not os.path.exists(csv_path):
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

def analyze_mae_by_time(X, folders_names=["lgbm", "xgb", "mlp", "lstm"]):
    WEEKEND = {5, 6}
    WEEKDAYS = {0, 1, 2, 3, 4}
    model_map = {
        "lgbm": "LightGBM",
        "xgb": "XGBoost",
        "mlp": "Daugiasluoksnis perceptronas",
        "lstm": "Ilgos trumpalaikės atminties tinklas"
    }
    WEEKDAY_NAMES = ["Pirmadienis", "Antradienis", "Trečiadienis",
                 "Ketvirtadienis", "Penktadienis", "Šeštadienis", "Sekmadienis"]
    hour_rows = []
    weekday_rows = []
    avg_rows = []

    for model in folders_names:
        df, _ = make_df(X, model)
        df["abs_error"] = df["error"].abs()
        model_name = model_map.get(model, model)

        for hour, mae in df.groupby("hour")["abs_error"].mean().items():
            hour_rows.append({"model": model_name, "hour": hour, "mae": round(mae, 4)})

        for day_idx, mae in df.groupby("weekday")["abs_error"].mean().items():
            weekday_rows.append({"model": model_name, "weekday": WEEKDAY_NAMES[day_idx], "mae": round(mae, 4)})

        avg_rows.append({
            "model":            model_name,
            "avg_weekday_mae":  round(df[df["weekday"].isin(WEEKDAYS)]["abs_error"].mean(), 4),
            "avg_weekend_mae":  round(df[df["weekday"].isin(WEEKEND)]["abs_error"].mean(), 4),
        })

    pd.DataFrame(hour_rows).pivot(index="hour", columns="model", values="mae").to_csv(os.path.join(BASE_DIR, "mae_by_hour.csv"))
    print(os.path.join(BASE_DIR, "mae_by_hour.csv"))

    pd.DataFrame(weekday_rows).pivot(index="weekday", columns="model", values="mae").reindex(WEEKDAY_NAMES).to_csv(os.path.join(BASE_DIR, "mae_by_weekday.csv"))
    print(os.path.join(BASE_DIR, "mae_by_weekday.csv"))

    pd.DataFrame(avg_rows).set_index("model").to_csv(os.path.join(BASE_DIR, "mae_weekday_vs_weekend.csv"))
    print(os.path.join(BASE_DIR, "mae_weekday_vs_weekend.csv"))