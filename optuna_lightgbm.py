import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score


print("LOADING DATA")
X_train = pd.read_csv('datasheets/X_train.csv')
X_test = pd.read_csv('datasheets/X_test.csv')
y_train = pd.read_csv('datasheets/y_train.csv').squeeze()  
y_test = pd.read_csv('datasheets/y_test.csv').squeeze()    

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

print("\n" + "=" * 70)
print("BASELINE MODELS")
print("=" * 70)

# Baseline 1: Last known value
y_pred_last = y_test.shift(1).fillna(y_train.iloc[-1])
baseline1_mae = mean_absolute_error(y_test, y_pred_last)
baseline1_default_rmse = np.sqrt(mean_squared_error(y_test, y_pred_last))
baseline1_default_r2 = r2_score(y_test, y_pred_last)
print(f"\nBaseline 1 (Last Value): MAE = {baseline1_mae:.4f}")
print(f"   RMSE: {baseline1_default_rmse:.4f}")
print(f"   R²:   {baseline1_default_r2:.4f}")

# Baseline 2: Mean of training data
y_pred_mean = np.full_like(y_test, y_train.mean())
baseline2_mae = mean_absolute_error(y_test, y_pred_mean)
print(f"Baseline 2 (Train Mean): MAE = {baseline2_mae:.4f}")
baseline2_default_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
baseline2_default_r2 = r2_score(y_test, y_pred_mean)
print(f"   RMSE: {baseline2_default_rmse:.4f}")
print(f"   R²:   {baseline2_default_r2:.4f}")

# Baseline 3: Rolling mean of last 24 hours
y_pred_rolling = X_test['rolling_mean_24']
baseline3_mae = mean_absolute_error(y_test, y_pred_rolling)
print(f"Baseline 3 (Rolling Mean): MAE = {baseline3_mae:.4f}")
baseline3_default_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rolling))
baseline3_default_r2 = r2_score(y_test, y_pred_rolling)
print(f"   RMSE: {baseline3_default_rmse:.4f}")
print(f"   R²:   {baseline3_default_r2:.4f}")

print("\n" + "=" * 70)
print("LIGHTGBM")
print("=" * 70)

lgb_default = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1 
)
lgb_default.fit(X_train, y_train)
y_pred_lgb_default = lgb_default.predict(X_test)

lgb_default_mae = mean_absolute_error(y_test, y_pred_lgb_default)
lgb_default_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb_default))
lgb_default_r2 = r2_score(y_test, y_pred_lgb_default)

print(f"\nLightGBM Performance:")
print(f"   MAE:  {lgb_default_mae:.4f}")
print(f"   RMSE: {lgb_default_rmse:.4f}")
print(f"   R²:   {lgb_default_r2:.4f}")
print(f"\n MAE improvements over baselines:\n")
print(f"   Baseline_1 improvement: {(1 - lgb_default_mae/baseline1_mae)*100:.1f}%")
print(f"   Baseline_2 improvement: {(1 - lgb_default_mae/baseline2_mae)*100:.1f}%")
print(f"   Baseline_3 improvement: {(1 - lgb_default_mae/baseline3_mae)*100:.1f}%")

lgb_default_fair = lgb.LGBMRegressor(
    objective="fair",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1 
)
lgb_default_fair.fit(X_train, y_train)
y_pred_lgb_default_fair = lgb_default_fair.predict(X_test)

lgb_default_fair_mae = mean_absolute_error(y_test, y_pred_lgb_default_fair)
lgb_default_fair_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb_default_fair))
lgb_default_fair_r2 = r2_score(y_test, y_pred_lgb_default_fair)

print(f"\nLightGBM Fair Performance:")
print(f"   MAE:  {lgb_default_fair_mae:.4f}")
print(f"   RMSE: {lgb_default_fair_rmse:.4f}")
print(f"   R²:   {lgb_default_fair_r2:.4f}")

lgb_default_huber = lgb.LGBMRegressor(
    objective="huber",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1 
)
lgb_default_huber.fit(X_train, y_train)
y_pred_lgb_default_huber = lgb_default_huber.predict(X_test)

lgb_default_huber_mae = mean_absolute_error(y_test, y_pred_lgb_default_huber)
lgb_default_huber_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb_default_huber))
lgb_default_huber_r2 = r2_score(y_test, y_pred_lgb_default_huber)

print(f"\nLightGBM Huber Performance:")
print(f"   MAE:  {lgb_default_huber_mae:.4f}")
print(f"   RMSE: {lgb_default_huber_rmse:.4f}")
print(f"   R²:   {lgb_default_huber_r2:.4f}")

for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
              1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]:
    lgb_huber = lgb.LGBMRegressor(
        objective="huber",
        alpha=alpha,
        n_estimators=100,
        random_state=42
    )
    lgb_huber.fit(X_train, y_train)
    y_pred = lgb_huber.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Huber loss with alpha={alpha}: MAE = {mae:.4f}")

c_values = [30, 35, 40, 45, 50, 55, 60]

for c in c_values:
    lgb_fair = lgb.LGBMRegressor(
        objective="fair",
        fair_c=c,  # Correct parameter for Fair loss
        n_estimators=100,
        random_state=42
    )
    lgb_fair.fit(X_train, y_train)
    y_pred = lgb_fair.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Fair loss with fair_c={c}: MAE = {mae:.4f}")

#optuna.logging.set_verbosity(optuna.logging.WARNING)
def objective(trial):
    dtrain = lgb.Dataset(X_train, label=y_train)
    
    loss_function = trial.suggest_categorical("loss_function", 
                                             ["regression", "regression_l1", "huber", "fair"])
    
    param = {
        "objective": loss_function, 
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "max_depth": trial.suggest_int("max_depth", -1, 50),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
    }
    
    if loss_function == "huber":
        param["alpha"] = trial.suggest_float("huber_alpha", 11, 15)
    elif loss_function == "fair":
        param["c"] = trial.suggest_float("fair_c", 30, 60)
    
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    return mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))
print("\nBest trial:")
trial = study.best_trial
print("  Value (MAE): {:.4f}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))