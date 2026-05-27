import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import shap
import pandas as pd
from plot_style import PlotStyle

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
    'rolling_mean_168': 'Slankusis vidurkis 168h',
    'rolling_std_168': 'Slankusis st. nuokrypis 168h',
}

STYLE = PlotStyle()
MAX_FEATURES_DISPLAYED = 5
BASE_DIR = base_dir = os.path.dirname(__file__)
X_label="Vidutinis požymio indėlis prie prognozės (EUR/MWh)"
Y_label="Duomenų požymiai\n(5 labiausiai prisidėję)"
Y_label_individual="Požymio indėlis prie prognozės (EUR/MWh)"
X_label_individual="Požymio reikšmė (EUR/MWh)"

def plot_importance(model, feature_names, X, output_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_array = shap_values.values
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df = X_df.rename(columns=FEATURE_NAMES)

    plt.figure()
    shap.summary_plot(shap_array, X_df, plot_type="bar", show=False, max_display = MAX_FEATURES_DISPLAYED)
    mean_shap = np.abs(shap_array).mean(axis=0)
    df = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": mean_shap
    }).sort_values("mean_abs_shap", ascending=False)
    csv_path = os.path.join(BASE_DIR, output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(csv_path)
    fig = plt.gcf()  
    ax = plt.gca()
    plt.xlabel(X_label,fontsize=STYLE.label_size)
    plt.ylabel(Y_label,fontsize=STYLE.label_size)
    path = os.path.join(BASE_DIR,output_dir, "shap_bar.png")
    STYLE.apply(fig,ax)
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches="tight")
    print(path)
    for feature in ["lag_1", "rolling_mean_24", "hour", "dayofweek","lag_6","lag_24","lag_12","dayofyear"]:
        idx = list(feature_names).index(feature)
        plt.figure()
        shap.dependence_plot(idx, shap_array, X_df, show=False, interaction_index=None)
        fig = plt.gcf()  
        ax = plt.gca()
        #fig.set_size_inches(11, 4)
        plt.ylabel(Y_label_individual,fontsize=STYLE.label_size)
        if feature == "hour":
            plt.xlabel("Dienos valanda",fontsize=STYLE.label_size)
            plt.xticks(range(24))
        elif feature == "dayofweek":
            plt.xlabel("Savaitės diena",fontsize=STYLE.label_size)
        elif feature == "dayofyear":
            plt.xlabel("Metų diena",fontsize=STYLE.label_size)
        else:
            plt.xlabel(X_label_individual,fontsize=STYLE.label_size)
        path = os.path.join(BASE_DIR, output_dir, f"shap_scatter_{feature}.png")
        STYLE.apply(fig,ax)
        plt.savefig(path, dpi=STYLE.dpi, bbox_inches="tight")
        print(path)

def plot_importance_mlp(model, feature_names, X, output_dir):
    X_values = X.to_numpy()
    rng = np.random.default_rng(1)
    indices = rng.choice(len(X_values), size=500, replace=False)
    background = X_values[indices[:100]]
    X_sample = X_values[indices[100:500]]
    predict_function = lambda x: model.predict(x).ravel()
    explainer = shap.KernelExplainer(predict_function, background)
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names).rename(columns=FEATURE_NAMES)
    shap_array = explainer.shap_values(X_sample) 
    plt.figure()
    shap.summary_plot(shap_array, X_sample_df, plot_type="bar", show=False, max_display =MAX_FEATURES_DISPLAYED)
    mean_shap = np.abs(shap_array).mean(axis=0)
    df = pd.DataFrame({
        "feature": X_sample_df.columns,
        "mean_abs_shap": mean_shap
    }).sort_values("mean_abs_shap", ascending=False)
    csv_path = os.path.join(BASE_DIR, output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(csv_path)
    fig = plt.gcf()  
    ax = plt.gca()
    plt.xlabel(X_label,fontsize=STYLE.label_size)
    plt.ylabel(Y_label,fontsize=STYLE.label_size)
    path = os.path.join(BASE_DIR, output_dir, "shap_bar_mlp.png")
    STYLE.apply(fig,ax)
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches="tight")
    print(path)
    
    for feature in ["lag_1", "rolling_mean_24", "hour", "dayofweek","lag_6","lag_24","lag_12","dayofyear"]:
        idx = list(feature_names).index(feature)
        plt.figure()
        shap.dependence_plot(idx, shap_array, X_sample_df, interaction_index=None, show=False)
        fig = plt.gcf()  
        ax = plt.gca()
        plt.ylabel(Y_label_individual,fontsize=STYLE.label_size)
        if feature == "hour":
            plt.xlabel("Dienos valanda",fontsize=STYLE.label_size)
        elif feature == "dayofweek":
            plt.xlabel("Savaitės diena",fontsize=STYLE.label_size)
        elif feature == "dayofyear":
            plt.xlabel("Metų diena",fontsize=STYLE.label_size)
        else:
            plt.xlabel(X_label_individual,fontsize=STYLE.label_size)
        path = os.path.join(BASE_DIR, output_dir, f"shap_scatter_{feature}_mlp.png")
        STYLE.apply(fig,ax)
        plt.savefig(path, dpi=STYLE.dpi, bbox_inches="tight")
        print(path)

def plot_importance_lstm(model, feature_names, X, output_dir, time_steps=6):
    
    def predict_fn(X_flat):
        X_3d = X_flat.reshape(-1, time_steps, len(feature_names))
        return model.predict(X_3d).ravel()
    
    n = len(X) - time_steps + 1
    indexes = np.arange(time_steps)[None, :] + np.arange(n)[:, None]
    X_values = X.to_numpy()
    X_seq = X_values[indexes]

    rng = np.random.default_rng(1)
    indices = rng.choice(len(X_seq), size=300, replace=False)
    background = X_seq[indices[:120]]
    X_sample   = X_seq[indices[120:300]] 

    background_flat = background.reshape(background.shape[0], -1)
    X_sample_flat   = X_sample.reshape(X_sample.shape[0], -1)

    explainer = shap.KernelExplainer(predict_fn, background_flat)
    shap_raw = explainer.shap_values(X_sample_flat)
    shap_3d = shap_raw.reshape(len(X_sample), time_steps, len(feature_names))
    max = np.max(X["lag_1"])
    min = np.min(X["lag_1"])
    price_range = max - min
    shap_by_feat = np.mean(np.abs(shap_3d), axis=1) * price_range
    shap_for_scatter = np.mean(shap_3d, axis=1) * price_range
    X_sample_last = X_sample[:, -1, :]
    X_sample_df   = pd.DataFrame(X_sample_last, columns=feature_names).rename(columns=FEATURE_NAMES)

    plt.figure()
    shap.summary_plot(shap_by_feat, X_sample_df, plot_type="bar", show=False, max_display=MAX_FEATURES_DISPLAYED)
    
    plt.xlabel(X_label,fontsize=STYLE.label_size)
    plt.ylabel(Y_label,fontsize=STYLE.label_size)
    fig = plt.gcf()  
    ax = plt.gca()
    STYLE.apply(fig,ax)
    path = os.path.join(BASE_DIR, output_dir, "shap_bar_lstm.png")
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches="tight")
    print(path)
    mean_shap = np.mean(shap_by_feat, axis=0)
    df = pd.DataFrame({
        "feature": X_sample_df.columns,
        "mean_abs_shap": mean_shap
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = os.path.join(BASE_DIR, output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(csv_path)
    for feature in ["lag_1", "rolling_mean_24", "hour", "dayofweek","lag_6","lag_24","lag_12","dayofyear","lag_48","lag_168"]:
        idx = list(feature_names).index(feature)
        plt.figure()
        shap.dependence_plot(idx, shap_for_scatter, X_sample_df, interaction_index=None, show=False)
        fig = plt.gcf()  
        ax = plt.gca()
        plt.ylabel(Y_label_individual,fontsize=STYLE.label_size)
        if feature == "hour":
            plt.xlabel("Dienos valanda",fontsize=STYLE.label_size)
        elif feature == "dayofweek":
            plt.xlabel("Savaitės diena",fontsize=STYLE.label_size)
        elif feature == "dayofyear":
            plt.xlabel("Metų diena",fontsize=STYLE.label_size)
        else:
            plt.xlabel(X_label_individual,fontsize=STYLE.label_size)
        STYLE.apply(fig,ax)
        path = os.path.join(BASE_DIR, output_dir, f"shap_scatter_{feature}.png")
        plt.savefig(path, dpi=STYLE.dpi, bbox_inches="tight")
        print(path)