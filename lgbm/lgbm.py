from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from .build_lightgbm import build_lightgbm

def lgbm(X, y, n_splits=5, test_size=0.2, objective='regression'):
    split_idx = int(len(X) * (1 - test_size))
    X_cv, X_test = X[:split_idx], X[split_idx:]
    y_cv, y_test = y[:split_idx], y[split_idx:]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_train, X_val = X_cv[train_idx], X_cv[val_idx]
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]

        model = build_lightgbm(n_estimators = 500, learning_rate = 0.01,
                                max_depth = 6, num_leaves = 32, feature_fraction=0.8, objective=objective)

        model.fit(X_train, y_train)
        
        valPredict = model.predict(X_val)

        fold_rmse = np.sqrt(mean_squared_error(valPredict, y_val))
        fold_rmses.append(fold_rmse)
        print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    model = build_lightgbm(n_estimators=500, learning_rate=0.01,
                           max_depth=6, num_leaves=32, feature_fraction=0.8, objective=objective)
    model.fit(X_cv, y_cv)

    testPredict = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(testPredict, y_test))
    print(f"Test RMSE: {test_rmse:.4f}")

    return {
        "fold_rmses": fold_rmses,
        "mean_cv_rmse": np.mean(fold_rmses),
        "test_rmse": test_rmse,
        "predictions": testPredict,
        "actuals": y_test,
        "model": model
    }