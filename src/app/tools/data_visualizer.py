import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_stock_prices(dataframe: pd.DataFrame, column: str = 'Close', title: str = 'Stock Prices Over Time'):
    """Plots stock prices over time."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dataframe, x=dataframe.index, y=column)
    plt.xticks(ticks=np.linspace(0, len(dataframe.index) - 1, 4, dtype=int), rotation=45)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Close Price')
    plt.show()

def plot_performance_metrics(train_mse: list, val_mse: list, train_r2: list, val_r2: list, epochs: int):
    """Plots Train and Validation MSE and R2 over epochs."""
    epochs_range = range(1, epochs + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot MSE
    sns.lineplot(ax=axes[0], x=epochs_range, y=train_mse, label='Train MSE')
    sns.lineplot(ax=axes[0], x=epochs_range, y=val_mse, label='Validation MSE')
    axes[0].set_title('Train and Validation MSE over Epochs')
    axes[0].set_ylabel('MSE')
    axes[0].legend()

    # Plot R2
    sns.lineplot(ax=axes[1], x=epochs_range, y=train_r2, label='Train R2')
    sns.lineplot(ax=axes[1], x=epochs_range, y=val_r2, label='Validation R2')
    axes[1].set_title('Train and Validation R2 over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('R2')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_predictions(actuals: np.ndarray, predictions: np.ndarray, title: str = 'Actual vs Predicted Stock Prices'):
    """Plots actual vs predicted stock prices."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(len(actuals)), y=actuals, label='Actual')
    sns.lineplot(x=range(len(predictions)), y=predictions, label='Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Stock Price')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_errors(errors: np.ndarray, title: str = 'Distribution of Prediction Errors (Residuals)'):
    """Plots the distribution of prediction errors."""
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.axvline(0, color='black', linestyle='--', label='Zero error')
    plt.title(title)
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_original_vs_filtered(original_df: pd.DataFrame, filtered_df: pd.DataFrame, column: str = 'Close', title: str = 'AMZN Stock Prices: Original vs Filtered'):
    """Plots original vs filtered stock prices."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=original_df, x=original_df.index, y=column, label='Original')
    sns.lineplot(data=filtered_df, x=filtered_df.index, y=column, label='Filtered')
    plt.xticks(ticks=np.linspace(0, len(original_df.index) - 1, 4, dtype=int), rotation=45)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def plot_enhanced_features(dataframe: pd.DataFrame):
    """Plots enhanced features of the stock data."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    tick_positions = np.linspace(0, len(dataframe.index) - 1, 4, dtype=int)

    # Plot Close
    sns.lineplot(ax=axes[0], data=dataframe, x=dataframe.index, y='Close', label='Close')
    axes[0].set_title('AMZN Closing Price')

    # RSI
    sns.lineplot(ax=axes[1], data=dataframe, x=dataframe.index, y='RSI', label='RSI', color='orange')
    axes[1].set_title('RSI')

    # Bandwidth
    sns.lineplot(ax=axes[2], data=dataframe, x=dataframe.index, y='Bandwidth', label='Bandwidth', color='green')
    axes[2].set_title('Bollinger Bandwidth')

    # %B
    sns.lineplot(ax=axes[3], data=dataframe, x=dataframe.index, y='%B', label='%B', color='purple')
    axes[3].set_title('Bollinger %B')

    for ax in axes:
        ax.set_xticks(dataframe.index[tick_positions])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
