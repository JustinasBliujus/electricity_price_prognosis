import xgboost

def build_xgboost(n_estimators, learning_rate, max_depth, colsample_bytree, objective):
    xgb_default = xgboost.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        colsample_bytree=colsample_bytree,
        objective=objective,
        random_state=42,
        n_jobs=-1,
    )
    return xgb_default