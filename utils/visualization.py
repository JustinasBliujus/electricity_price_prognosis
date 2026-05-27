import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.graphics.tsaplots import plot_acf
from plot_style import PlotStyle
import numpy as np

STYLE = PlotStyle()

def plot_values(df,folder_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['utc'],df['value'])
    index = np.linspace(0, len(df) - 1, 5, dtype=int)
    x_range = df['utc']
    ax.set_xticks(x_range.iloc[index])
    ax.set_xticklabels(x_range.iloc[index].dt.strftime("%Y-%m-%d"))
    ax.set_xlabel('Data', fontsize=STYLE.label_size)
    ax.set_ylabel('Kaina (EUR/MWh)', fontsize=STYLE.label_size)
    STYLE.apply(fig,ax)
    path = os.path.join(folder_path, 'values.png')
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)
    
def plot_autocorrelation(df, folder_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(df['value'], lags=168, ax=ax)
    
    ax.set_xlabel('Poslinkis (valandomis)', fontsize=STYLE.label_size)
    ax.set_ylabel('Autokoreliacija', fontsize=STYLE.label_size)
    
    ax.set_xticks(range(0, 169, 12))
    ax.set_title('')
    ax.set_ylim(bottom=-0.1)
    
    STYLE.apply(fig,ax)
    
    path = os.path.join(folder_path, 'autocorrelation.png')
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)

def plot_price_distribution(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_ml['value'], bins=50, alpha=1,edgecolor='black')
    ax.set_xlabel('Kaina (EUR/MWh)', fontsize=STYLE.label_size)
    ax.set_ylabel('Dažnis (kiek kainų patenka)',fontsize=STYLE.label_size)
    path = os.path.join(folder_path, 'distribution.png')
    STYLE.apply(fig,ax)
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)    

def plot_price_by_hour(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    df_ml.boxplot(column='value', by='hour', ax=ax) #1.5 * IQR 
    ax.set_title('')
    ax.set_xlabel('Valanda', fontsize=STYLE.label_size)
    ax.set_ylabel('Kaina (EUR/MWh)', fontsize=STYLE.label_size)
    plt.suptitle('')
    STYLE.apply(fig,ax)
    path = os.path.join(folder_path, 'boxplot_hour.png')
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)

def plot_price_by_month(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    df_ml.boxplot(column='value', by='month', ax=ax)
    ax.set_title('')
    ax.set_xlabel('Mėnesis', fontsize=STYLE.label_size)
    ax.set_ylabel('Kaina (EUR/MWh)', fontsize=STYLE.label_size)
    plt.suptitle('')
    STYLE.apply(fig,ax)
    path=os.path.join(folder_path, 'boxplot_month.png')
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)

def plot_avg_price_by_day(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_by_day = df_ml.groupby('dayofweek')['value'].mean()
    days = ['P', 'A', 'T', 'K', 'P', 'Š', 'S']
    ax.bar(days, avg_by_day.values, color='skyblue')
    ax.set_title('')
    ax.set_xlabel('Savaitės diena', fontsize=STYLE.label_size)
    ax.set_ylabel('Vidutinė kaina (EUR/MWh)', fontsize=STYLE.label_size)
    STYLE.apply(fig,ax)
    path=os.path.join(folder_path, 'avg_by_day.png')
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)

def plot_avg_price_by_month(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_by_month = df_ml.groupby('month')['value'].mean()
    months = ['S', 'V', 'K', 'B', 'G', 'B', 'L', 'Rupj', 'Rugs', 'S', 'L', 'G']
    ax.bar(months, avg_by_month.values, color='skyblue')
    ax.set_xlabel('Mėnuo', fontsize=STYLE.label_size)
    ax.set_ylabel('Vidutinė kaina (EUR/MWh)', fontsize=STYLE.label_size)
    STYLE.apply(fig,ax)
    path=os.path.join(folder_path, 'avg_by_month.png')
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)

def plot_avg_month_and_weekday(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(14, 5))

    months = ['Sausis', 'Vasaris', 'Kovas', 'Balandis', 'Gegužė', 'Birželis', 'Liepa', 'Rugpjūtis', 'Rugsėjis', 'Spalis', 'Lapkritis', 'Gruodis']
    avg_weekday = df_ml[df_ml['weekend'] == 0].groupby('month')['value'].mean()
    avg_weekend = df_ml[df_ml['weekend'] == 1].groupby('month')['value'].mean()

    x = range(len(months))
    width = 0.4

    ax.bar([i - width/2 for i in x], avg_weekday.values, width=width, color='skyblue',edgecolor='black',label='Darbo diena')
    ax.bar([i + width/2 for i in x], avg_weekend.values, width=width, color='salmon',edgecolor='black',label='Savaitgalis')

    ax.set_xticks(list(x))
    ax.set_xticklabels(months)
    ax.set_xlabel('Mėnuo', fontsize=STYLE.label_size)
    ax.set_ylabel('Vidutinė kaina (EUR/MWh)', fontsize=STYLE.label_size)
    STYLE.apply(fig,ax)
    path=os.path.join(folder_path, 'avg_month_weekday.png')
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches='tight')
    print(path)

def plot_target_correlations(df_ml, X, folder_path):
    all_features_df = pd.concat([df_ml[['value']], X], axis=1)
    corr_all = all_features_df.corr()
    
    corr_with_value = pd.DataFrame({
        'feature': corr_all.index,
        'correlation_with_value': corr_all['value'].values
    }).sort_values('correlation_with_value')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in corr_with_value['correlation_with_value']]
    bars = ax.barh(corr_with_value['feature'], corr_with_value['correlation_with_value'],color=colors, alpha=1)
    
    ax.axvline(x=0, color='black', linestyle='-')
    ax.set_xlabel('Koreliacija su kaina', fontsize=STYLE.label_size)
    ax.set_ylabel('Požymiai', fontsize=STYLE.label_size)
    ax.set_title('')
    for bar, val in zip(bars, corr_with_value['correlation_with_value']):
        ax.text(val + (0.01 if val >= 0 else -0.07), bar.get_y() + bar.get_height()/2,val, va='center', fontsize=9)
    
    STYLE.apply(fig,ax)
    path=os.path.join(folder_path, 'target_correlations.png')
    plt.savefig(os.path.join(folder_path, 'target_correlations.png'), dpi=STYLE.dpi, bbox_inches='tight')
    print(path)

def plot_hour_differences(df, folder_path):
    df = df.copy()
    indexes = np.linspace(0, len(df) - 1, 5, dtype=int)
    df["utc"] = pd.to_datetime(df["utc"])
    x_range = df['utc']
    df["date"] = df["utc"].dt.date
    df["hour"] = df["utc"].dt.hour
    filtered = df[df["hour"].isin([11, 17])]
    needed_hours = filtered.pivot(index="date", columns="hour", values="value")
    needed_hours = needed_hours.dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xticks(x_range.iloc[indexes])
    ax.set_xticklabels(x_range.iloc[indexes].dt.strftime("%Y-%m-%d"))
    ax.plot(needed_hours.index, needed_hours[11], label="11 valandos kainos")
    ax.plot(needed_hours.index, needed_hours[17], label="17 valandos kainos")
    ax.set_xlabel('Data', fontsize=STYLE.label_size)
    ax.set_ylabel('Kaina (EUR/MWh)', fontsize=STYLE.label_size)
    STYLE.apply(fig, ax)
    path = os.path.join(folder_path, "midday_evening.png")
    plt.savefig(path, dpi=STYLE.dpi, bbox_inches="tight")
    plt.close()
    print(path)

def generate_all_plots(df_ml, X):
    folder_path = os.path.join(os.path.dirname(__file__), "visualizations")
    plot_values(df_ml,folder_path)
    plot_hour_differences(df_ml,folder_path)
    plot_autocorrelation(df_ml,folder_path)
    plot_price_distribution(df_ml, folder_path)
    plot_price_by_hour(df_ml, folder_path)
    plot_price_by_month(df_ml, folder_path)
    plot_avg_price_by_day(df_ml, folder_path)
    plot_avg_price_by_month(df_ml,folder_path)
    plot_avg_month_and_weekday(df_ml,folder_path)
    plot_target_correlations(df_ml, X, folder_path)