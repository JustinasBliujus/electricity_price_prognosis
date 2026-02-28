import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def build_default_xgboost_model(X_train, y_train, X_test, y_test):
    print("\nLoading data for XGBoost")
    X_train = pd.read_csv('datasheets/X_train.csv')
    X_test = pd.read_csv('datasheets/X_test.csv')
    y_train = pd.read_csv('datasheets/y_train.csv').squeeze()  
    y_test = pd.read_csv('datasheets/y_test.csv').squeeze()    

    xgb_default = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    xgb_default.fit(X_train, y_train)
    y_pred_xgb_default = xgb_default.predict(X_test)

    xgb_default_mae = mean_absolute_error(y_test, y_pred_xgb_default)
    xgb_default_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_default))
    xgb_default_r2 = r2_score(y_test, y_pred_xgb_default)

    print(f"\nXGBoost Performance:")
    print(f"   MAE:  {xgb_default_mae:.4f}")
    print(f"   RMSE: {xgb_default_rmse:.4f}")
    print(f"   R²:   {xgb_default_r2:.4f}")
