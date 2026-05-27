import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_baseline(y_test, y_train):
    y_test = np.array(y_test)
    y_train = np.array(y_train)

    full = np.concatenate([y_train, y_test])

    mae_total = 0
    rmse_total = 0
    prediction_history = []
    actual_history = []

    n_windows = len(y_test)-24

    for i in range(n_windows):
        start = len(y_train) + i

        predicted = full[start - 24:start]
        actual = y_test[i:i + 24]

        mae_total += mean_absolute_error(actual, predicted)
        rmse_total += root_mean_squared_error(actual, predicted)

        prediction_history.append(predicted)
        actual_history.append(actual)

    mae = mae_total / len(prediction_history)
    rmse = rmse_total / len(prediction_history)

    print(f"baseline MAE: {mae}, RMSE: {rmse}")

    df = pd.DataFrame(prediction_history)
    df.to_csv(os.path.join(CURRENT_DIR, "baseline_predictions.csv"), index=False)