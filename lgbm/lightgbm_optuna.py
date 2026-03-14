import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from electricity_price_prognosis.lightgbm.build_lightgbm import build_lightgbm

def lightgbm_optuna(X, y, n_splits=5, test_size=0.2, n_trials=100):
    split_idx = int(len(X) * (1 - test_size))
    X_cv, X_test = X[:split_idx], X[split_idx:]
    y_cv, y_test = y[:split_idx], y[split_idx:]

    def objective(trial):
        n_estimators     = trial.suggest_int("n_estimators", 100, 1000, step=100)
        learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        max_depth        = trial.suggest_int("max_depth", 3, 10)
        num_leaves       = trial.suggest_int("num_leaves", 15, 63)
        feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_rmses = []

        for train_idx, val_idx in tscv.split(X_cv):
            X_train, X_val = X_cv[train_idx], X_cv[val_idx]
            y_train, y_val = y_cv[train_idx], y_cv[val_idx]

            model = build_lightgbm(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                feature_fraction=feature_fraction
            )

            model.fit(X_train, y_train)

            pred = model.predict(X_val)
            fold_rmses.append(np.sqrt(mean_squared_error(y_val, pred)))

        return np.mean(fold_rmses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best params:", study.best_params)
    print(f"Best CV RMSE: {study.best_value:.4f}")

    p = study.best_params

    best_model = build_lightgbm(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        num_leaves=p["num_leaves"],
        feature_fraction=p["feature_fraction"]
    )
    best_model.fit(X_cv, y_cv)

    testPredict = best_model.predict(X_test)
    test_rmse   = np.sqrt(mean_squared_error(y_test, testPredict))
    print(f"Test RMSE: {test_rmse:.4f}")

    pd.DataFrame({
        "actual":    y_test.flatten(),
        "predicted": testPredict.flatten(),
        "error":     (y_test - testPredict).flatten()
    }).to_csv("predictions.csv", index=False)

    study.trials_dataframe().to_csv("optuna_trials.csv", index=False)

    pd.DataFrame([{**study.best_params, "cv_rmse": study.best_value,
                   "test_rmse": test_rmse}]).to_csv("best_params.csv", index=False)

    return {
        "best_params": study.best_params,
        "cv_rmse":     study.best_value,
        "test_rmse":   test_rmse,
        "predictions": testPredict,
        "actuals":     y_test,
        "model":       best_model,
        "study":       study
    }