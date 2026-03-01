import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def setup_plotting_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

def plot_train_val_test_split(df_ml, y_train, y_test, y_val, folder_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    train_dates = df_ml['utc'].iloc[:len(y_train)]
    val_dates = df_ml['utc'].iloc[len(y_train):len(y_train)+len(y_val)]
    test_dates = df_ml['utc'].iloc[len(y_train)+len(y_val):]
    
    ax.plot(train_dates, y_train.values, label='Train', linewidth=1, alpha=0.7)
    ax.plot(val_dates, y_val.values, label='Validation', linewidth=1, alpha=0.7)
    ax.plot(test_dates, y_test.values, label='Test', linewidth=1, alpha=0.7)
    ax.axvline(x=train_dates.iloc[-1], color='red', linestyle='--', alpha=0.5, label='Train/Validation Split')
    ax.axvline(x=val_dates.iloc[-1], color='orange', linestyle='--', alpha=0.5, label='Validation/Test Split')
    
    ax.set_title('Train/Validation/Test Split', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'train_val_test_split.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_rolling_statistics(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_ml['utc'], df_ml['value'], label='Original', linewidth=1, alpha=0.5)
    ax.plot(df_ml['utc'], df_ml['rolling_mean_24'], label='24h Rolling Mean', linewidth=1.5)
    ax.plot(df_ml['utc'], df_ml['rolling_std_24'], label='24h Rolling Std', linewidth=1)
    
    ax.set_title('Rolling Statistics (24h window)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'rolling_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_price_distribution(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_ml['value'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Electricity Prices', fontsize=12, fontweight='bold')
    ax.set_xlabel('Price (EUR/MWh)')
    ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_price_by_hour(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    df_ml.boxplot(column='value', by='hour', ax=ax) #1.5 * IQR ?
    ax.set_title('Price Distribution by Hour', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price (EUR/MWh)')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'boxplot_hour.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_price_by_month(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    df_ml.boxplot(column='value', by='month', ax=ax)
    ax.set_title('Price Distribution by Month', fontsize=12, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Price (EUR/MWh)')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'boxplot_month.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_avg_price_by_day(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_by_day = df_ml.groupby('dayofweek')['value'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    ax.bar(days, avg_by_day.values, color='skyblue', edgecolor='black')
    ax.set_title('Average Price by Day of Week', fontsize=12, fontweight='bold')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Price (EUR/MWh)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'avg_by_day.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_weekday_vs_weekend(df_ml, folder_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    weekend_data = [df_ml[df_ml['weekend'] == 0]['value'], 
                    df_ml[df_ml['weekend'] == 1]['value']]
    
    ax.boxplot(weekend_data, tick_labels=['Weekday', 'Weekend'])
    ax.set_title('Weekday vs Weekend Prices', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (EUR/MWh)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'weekday_weekend.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df_ml, X, folder_path):
    all_features_df = pd.concat([df_ml[['value']], X], axis=1)
    corr_all = all_features_df.corr()
    n = len(corr_all.columns)
    
    fig, ax = plt.subplots(figsize=(n * 0.4, n * 0.3))
    sns.heatmap(corr_all, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True, 
                linewidths=0.3, 
                cbar_kws={"shrink": 0.5},
                annot_kws={'size': 7},
                ax=ax)
    
    ax.set_title('Full Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', labelsize=8, rotation=90)
    ax.tick_params(axis='y', labelsize=8, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'full_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_target_correlations(df_ml, X, folder_path):
    all_features_df = pd.concat([df_ml[['value']], X], axis=1)
    corr_all = all_features_df.corr()
    
    corr_with_value = pd.DataFrame({
        'feature': corr_all.index,
        'correlation_with_value': corr_all['value'].values
    }).sort_values('correlation_with_value')
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(corr_with_value) * 0.3)))
    colors = ['red' if x < 0 else 'green' for x in corr_with_value['correlation_with_value']]
    bars = ax.barh(corr_with_value['feature'], corr_with_value['correlation_with_value'],
                   color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Correlation with Price (value)', fontsize=12)
    ax.set_title('Feature Correlation with Target Variable (value)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, corr_with_value['correlation_with_value']):
        ax.text(val + (0.01 if val >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'target_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(df_ml, X, y_train, y_test, X_val, y_val, folder_path):
    print("\nGenerating plots:")
    setup_plotting_style()
    
    plot_train_val_test_split(df_ml, y_train, y_test, y_val, folder_path)
    plot_rolling_statistics(df_ml, folder_path)
    plot_price_distribution(df_ml, folder_path)
    plot_price_by_hour(df_ml, folder_path)
    plot_price_by_month(df_ml, folder_path)
    plot_avg_price_by_day(df_ml, folder_path)
    plot_weekday_vs_weekend(df_ml, folder_path)
    plot_correlation_matrix(df_ml, X, folder_path)
    plot_target_correlations(df_ml, X, folder_path)

    print("All plots generated and saved.")