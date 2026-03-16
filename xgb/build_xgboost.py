import xgboost

def build_xgboost(n_estimators, learning_rate, max_depth, colsample_bytree, objective,huber_slope=4):
    xgb_default = xgboost.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        colsample_bytree=colsample_bytree,
        objective=objective,
        huber_slope=huber_slope,
        random_state=42,
        n_jobs=-1,
    )
    return xgb_default