import lightgbm as lgbm

def build_lightgbm(n_estimators, learning_rate, max_depth, num_leaves, 
                   feature_fraction, objective='regression', alpha=15, fair_c=7):
    lgbm_default = lgbm.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        feature_fraction=feature_fraction,
        objective=objective,
        alpha=alpha,
        fair_c=fair_c,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    return lgbm_default