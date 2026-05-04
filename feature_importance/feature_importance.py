import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import shap
import pandas as pd

FEATURE_NAMES = {
    'hour': 'Valanda',
    'day': 'Diena',
    'month': 'Mėnuo',
    'year': 'Metai',
    'dayofweek': 'Savaitės diena',
    'quarter': 'Ketvirtis',
    'dayofyear': 'Metų diena',
    'weekend': 'Savaitgalis',
    'lag_1': 'Atsilikimas 1h',
    'lag_2': 'Atsilikimas 2h',
    'lag_3': 'Atsilikimas 3h',
    'lag_6': 'Atsilikimas 6h',
    'lag_12': 'Atsilikimas 12h',
    'lag_24': 'Atsilikimas 24h',
    'lag_48': 'Atsilikimas 48h',
    'lag_96': 'Atsilikimas 96h',
    'lag_168': 'Atsilikimas 168h',
    'rolling_mean_6': 'Slankusis vidurkis 6h',
    'rolling_std_6': 'Slankusis st. nuokrypis 6h',
    'rolling_mean_12': 'Slankusis vidurkis 12h',
    'rolling_std_12': 'Slankusis st. nuokrypis 12h',
    'rolling_mean_24': 'Slankusis vidurkis 24h',
    'rolling_std_24': 'Slankusis st. nuokrypis 24h',
    'rolling_mean_48': 'Slankusis vidurkis 48h',
    'rolling_std_48': 'Slankusis st. nuokrypis 48h',
    'rolling_mean_96': 'Slankusis vidurkis 96h',
    'rolling_std_96': 'Slankusis st. nuokrypis 96h',
    'rolling_mean_168': 'Slankusis vidurkis 168h',
    'rolling_std_168': 'Slankusis st. nuokrypis 168h',
}

def plot_importance(model, feature_names, X, output_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_array = shap_values.values
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df = X_df.rename(columns=FEATURE_NAMES)

    plt.figure()
    shap.summary_plot(shap_array, X_df, plot_type="bar", show=False, max_display = 5)
    plt.xlabel("Vidutinė SHAP reikšmė")
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    for feature in ["lag_1", "hour", "rolling_mean_24"]:
        idx = list(feature_names).index(feature)
        vertimas = FEATURE_NAMES.get(feature, feature)
        plt.figure()
        shap.dependence_plot(idx, shap_array, X_df, show=False, interaction_index=None)
        plt.ylabel(f"SHAP reikšmė")
        #plt.gcf().axes[-1].set_ylabel("")
        plt.savefig(os.path.join(output_dir, f"shap_scatter_{feature}.png"), dpi=150, bbox_inches="tight")
        plt.close()

def plot_importance_mlp(model, feature_names, X, output_dir):
    background = shap.sample(X, 500)
    
    predict_fn = lambda x: model.predict(x).ravel()
    explainer = shap.KernelExplainer(predict_fn, background)

    X_sample = shap.sample(X, 1200)
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names).rename(columns=FEATURE_NAMES)
    
    shap_array = explainer.shap_values(X_sample)  

    plt.figure()
    shap.summary_plot(shap_array, X_sample_df, plot_type="bar", show=False, max_display = 5)
    plt.xlabel("Vidutinė SHAP reikšmė")
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    for feature in ["lag_1", "dayofyear", "rolling_mean_24"]:
        idx = list(feature_names).index(feature)
        plt.figure()
        shap.dependence_plot(idx, shap_array, X_sample_df, interaction_index=None, show=False)
        plt.ylabel(f"SHAP reikšmė")
        plt.savefig(os.path.join(output_dir, f"shap_scatter_{feature}.png"), dpi=150, bbox_inches="tight")
        plt.close()

def plot_importance_lstm(model, feature_names, X, output_dir, time_steps=6):
    n = len(X) - time_steps + 1
    idx = np.arange(time_steps)[None, :] + np.arange(n)[:, None]
    X_values = X.to_numpy()
    X_seq = X_values[idx]

    rng = np.random.default_rng(42)

    indices = rng.choice(len(X_seq), size=300, replace=False)

    background = X_seq[indices[:120]]
    X_sample   = X_seq[indices[120:300]] 

    background_flat = background.reshape(background.shape[0], -1)
    X_sample_flat   = X_sample.reshape(X_sample.shape[0], -1)

    def predict_fn(X_flat):
        X_3d = X_flat.reshape(-1, time_steps, len(feature_names))
        return model.predict(X_3d).ravel()

    explainer  = shap.KernelExplainer(predict_fn, background_flat)
    shap_raw   = explainer.shap_values(X_sample_flat)
    shap_3d      = shap_raw.reshape(len(X_sample), time_steps, len(feature_names))
    max = np.max(X["lag_1"])
    min = np.min(X["lag_1"])
    price_range = max - min
    shap_by_feat = np.mean(np.abs(shap_3d), axis=1) * price_range
    shap_for_scatter = np.mean(shap_3d, axis=1)  * price_range
    X_sample_last = X_sample[:, -1, :]
    X_sample_df   = pd.DataFrame(X_sample_last, columns=feature_names).rename(columns=FEATURE_NAMES)

    plt.figure()
    shap.summary_plot(shap_by_feat, X_sample_df, plot_type="bar", show=False, max_display=5)
    plt.xlabel("Vidutinė SHAP reikšmė")
    plt.savefig(os.path.join(output_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    for feature in ["lag_1", "lag_3", "lag_24", "lag_6"]:
        idx = list(feature_names).index(feature)
        plt.figure()
        shap.dependence_plot(idx, shap_for_scatter, X_sample_df, interaction_index=None, show=False)
        plt.ylabel(f"SHAP reikšmė")
        plt.savefig(os.path.join(output_dir, f"shap_scatter_{feature}.png"), dpi=150, bbox_inches="tight")
        plt.close()