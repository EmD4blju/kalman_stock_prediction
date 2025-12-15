"""Nodes for model evaluation pipeline."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from typing import Dict, Any, Tuple
from logging import getLogger

from ...models.lstm_model import LSTMStockModel
from ...models.dataset import StockDataset

log = getLogger(__name__)


def _create_actual_vs_predicted_plot(actuals: np.ndarray, predictions: np.ndarray, title: str = 'Actual vs Predicted Stock Prices') -> plt.Figure:
    """
    Create actual vs predicted plot.
    
    Args:
        actuals: Actual values
        predictions: Predicted values
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=range(len(actuals)), y=actuals, label='Actual', ax=ax)
    sns.lineplot(x=range(len(predictions)), y=predictions, label='Predicted', ax=ax)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Stock Price ($)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _create_error_distribution_plot(errors: np.ndarray, title: str = 'Distribution of Prediction Errors (Residuals)') -> plt.Figure:
    """
    Create error distribution plot.
    
    Args:
        errors: Prediction errors
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True, ax=ax)
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero error')
    ax.set_title(title)
    ax.set_xlabel('Error (Actual - Predicted) ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def evaluate_base_model(
    test_dataset: pd.DataFrame,
    model: LSTMStockModel,
    scaler_y: Any,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], plt.Figure, plt.Figure]:
    """
    Evaluate base model on test dataset.
    
    Args:
        test_dataset: Test dataset
        model: Trained LSTM model
        scaler_y: Scaler for target variable
        parameters: Evaluation parameters
    
    Returns:
        Tuple containing:
            - Dictionary with evaluation metrics
            - Actual vs Predicted plot
            - Error distribution plot
    """
    log.info("Starting base model evaluation")
    
    # Extract parameters
    batch_size = parameters.get('batch_size', 10)
    target_column = parameters.get('target_column', 'Close')
    
    # Create dataset and loader
    test_stock_dataset = StockDataset(test_dataset, ticker='AMZN', target_column=target_column)
    test_loader = DataLoader(test_stock_dataset, batch_size=batch_size, shuffle=False)
    sequence_length = test_stock_dataset.X.shape[1]
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        sequence_length=sequence_length
    )
    
    # Calculate metrics
    test_mse = mean_squared_error(actuals, predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(actuals, predictions)
    test_r2 = r2_score(actuals, predictions)
    
    metrics = {
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'model_type': 'base'
    }
    
    # Create plots
    actual_vs_predicted_plot = _create_actual_vs_predicted_plot(
        actuals, predictions, title='Base Model: Actual vs Predicted Stock Prices'
    )
    error_distribution_plot = _create_error_distribution_plot(
        errors, title='Base Model: Distribution of Prediction Errors'
    )
    
    log.info(f"Base model evaluation completed. Test RMSE: ${metrics['test_rmse']:.2f}, Test R2: {metrics['test_r2']:.4f}")
    
    return metrics, actual_vs_predicted_plot, error_distribution_plot


def evaluate_enriched_model(
    test_dataset: pd.DataFrame,
    model: LSTMStockModel,
    scaler_y: Any,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], plt.Figure, plt.Figure]:
    """
    Evaluate enriched model on test dataset.
    
    Args:
        test_dataset: Test dataset with enriched features
        model: Trained LSTM model
        scaler_y: Scaler for target variable
        parameters: Evaluation parameters
    
    Returns:
        Tuple containing:
            - Dictionary with evaluation metrics
            - Actual vs Predicted plot
            - Error distribution plot
    """
    log.info("Starting enriched model evaluation")
    
    # Extract parameters
    batch_size = parameters.get('batch_size', 10)
    target_column = parameters.get('target_column', 'Close')
    
    # Create dataset and loader
    test_stock_dataset = StockDataset(test_dataset, ticker='AMZN_ENHANCED', target_column=target_column)
    test_loader = DataLoader(test_stock_dataset, batch_size=batch_size, shuffle=False)
    # For enriched model: input_dim=4, so sequence_length = total_features / 4
    sequence_length = test_stock_dataset.X.shape[1] // 4
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        sequence_length=sequence_length
    )
    
    # Calculate metrics
    test_mse = mean_squared_error(actuals, predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(actuals, predictions)
    test_r2 = r2_score(actuals, predictions)
    
    metrics = {
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'model_type': 'enriched'
    }
    
    # Create plots
    actual_vs_predicted_plot = _create_actual_vs_predicted_plot(
        actuals, predictions, title='Enriched Model: Actual vs Predicted Stock Prices'
    )
    error_distribution_plot = _create_error_distribution_plot(
        errors, title='Enriched Model: Distribution of Prediction Errors'
    )
    
    log.info(f"Enriched model evaluation completed. Test RMSE: ${metrics['test_rmse']:.2f}, Test R2: {metrics['test_r2']:.4f}")
    
    return metrics, actual_vs_predicted_plot, error_distribution_plot


def evaluate_kalman_model_filtered(
    test_dataset: pd.DataFrame,
    model: LSTMStockModel,
    scaler_y: Any,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], plt.Figure, plt.Figure]:
    """
    Evaluate Kalman model on filtered test dataset.
    
    Args:
        test_dataset: Test dataset with Kalman-filtered features
        model: Trained LSTM model
        scaler_y: Scaler for target variable
        parameters: Evaluation parameters
    
    Returns:
        Tuple containing:
            - Dictionary with evaluation metrics
            - Actual vs Predicted plot
            - Error distribution plot
    """
    log.info("Starting Kalman model evaluation on filtered data")
    
    # Extract parameters
    batch_size = parameters.get('batch_size', 10)
    target_column = parameters.get('target_column', 'Close')
    
    # Create dataset and loader
    test_stock_dataset = StockDataset(test_dataset, ticker='AMZN_KALMAN', target_column=target_column)
    test_loader = DataLoader(test_stock_dataset, batch_size=batch_size, shuffle=False)
    sequence_length = test_stock_dataset.X.shape[1]
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        sequence_length=sequence_length
    )
    
    # Calculate metrics
    test_mse = mean_squared_error(actuals, predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(actuals, predictions)
    test_r2 = r2_score(actuals, predictions)
    
    metrics = {
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'model_type': 'kalman_filtered',
        'evaluation_data': 'filtered'
    }
    
    # Create plots
    actual_vs_predicted_plot = _create_actual_vs_predicted_plot(
        actuals, predictions, title='Kalman Model (Filtered): Actual vs Predicted Stock Prices'
    )
    error_distribution_plot = _create_error_distribution_plot(
        errors, title='Kalman Model (Filtered): Distribution of Prediction Errors'
    )
    
    log.info(f"Kalman model (filtered) evaluation completed. Test RMSE: ${metrics['test_rmse']:.2f}, Test R2: {metrics['test_r2']:.4f}")
    
    return metrics, actual_vs_predicted_plot, error_distribution_plot


def evaluate_kalman_model_original(
    test_dataset: pd.DataFrame,
    model: LSTMStockModel,
    scaler_y: Any,
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], plt.Figure, plt.Figure]:
    """Evaluate Kalman model on original test data."""
    log.info("Starting Kalman model evaluation on original data")
    
    batch_size = parameters.get('batch_size', 10)
    target_column = parameters.get('target_column', 'Close')
    
    test_stock_dataset = StockDataset(test_dataset, ticker='AMZN', target_column=target_column)
    test_loader = DataLoader(test_stock_dataset, batch_size=batch_size, shuffle=False)
    sequence_length = test_stock_dataset.X.shape[1]
    
    loss_function = MSELoss()
    
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        sequence_length=sequence_length
    )
    
    test_mse = mean_squared_error(actuals, predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(actuals, predictions)
    test_r2 = r2_score(actuals, predictions)
    
    metrics = {
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'model_type': 'kalman_original'
    }
    
    actual_vs_predicted_plot = _create_actual_vs_predicted_plot(
        actuals, predictions, title='Kalman Model (Original): Actual vs Predicted Stock Prices'
    )
    error_distribution_plot = _create_error_distribution_plot(
        errors, title='Kalman Model (Original): Distribution of Prediction Errors'
    )
    
    log.info(f"Kalman model (original) evaluation completed. Test RMSE: ${metrics['test_rmse']:.2f}, Test R2: {metrics['test_r2']:.4f}")
    
    return metrics, actual_vs_predicted_plot, error_distribution_plot


def create_learning_curves_plot(
    train_metrics: Dict[str, Any],
    model_name: str
) -> plt.Figure:
    """
    Create learning curves plot from training metrics.
    
    Args:
        train_metrics: Dictionary containing training metrics history
        model_name: Name of the model for plot title
    
    Returns:
        Matplotlib figure with learning curves
    """
    log.info(f"Creating learning curves plot for {model_name}")
    
    epochs = train_metrics.get('epochs', len(train_metrics.get('train_mse_history', [])))
    train_mse = train_metrics.get('train_mse_history', [])
    val_mse = train_metrics.get('val_mse_history', [])
    train_r2 = train_metrics.get('train_r2_history', [])
    val_r2 = train_metrics.get('val_r2_history', [])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot MSE
    epochs_range = range(1, len(train_mse) + 1)
    axes[0].plot(epochs_range, train_mse, label='Training MSE', marker='o', linewidth=2)
    axes[0].plot(epochs_range, val_mse, label='Validation MSE', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title(f'{model_name}: Learning Curves - MSE', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # Log scale for better visualization
    
    # Plot R²
    axes[1].plot(epochs_range, train_r2, label='Training R²', marker='o', linewidth=2)
    axes[1].plot(epochs_range, val_r2, label='Validation R²', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title(f'{model_name}: Learning Curves - R²', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    log.info(f"Learning curves plot created for {model_name}")
    return fig


def plot_base_model_learning_curves(
    train_metrics: Dict[str, Any]
) -> plt.Figure:
    """
    Create learning curves plot for base model.
    
    Args:
        train_metrics: Training metrics dictionary
    
    Returns:
        Matplotlib figure
    """
    return create_learning_curves_plot(train_metrics, "Base Model")


def plot_enriched_model_learning_curves(
    train_metrics: Dict[str, Any]
) -> plt.Figure:
    """
    Create learning curves plot for enriched model.
    
    Args:
        train_metrics: Training metrics dictionary
    
    Returns:
        Matplotlib figure
    """
    return create_learning_curves_plot(train_metrics, "Enriched Model")


def plot_kalman_model_learning_curves(
    train_metrics: Dict[str, Any]
) -> plt.Figure:
    """
    Create learning curves plot for Kalman model.
    
    Args:
        train_metrics: Training metrics dictionary
    
    Returns:
        Matplotlib figure
    """
    return create_learning_curves_plot(train_metrics, "Kalman Model")
    """
    Evaluate Kalman model on original (unfiltered) test dataset.
    
    Args:
        test_dataset: Test dataset with original unfiltered features
        model: Trained LSTM model
        scaler_y: Scaler for target variable
        parameters: Evaluation parameters
    
    Returns:
        Tuple containing:
            - Dictionary with evaluation metrics
            - Actual vs Predicted plot
            - Error distribution plot
    """
    log.info("Starting Kalman model evaluation on original data")
    
    # Extract parameters
    batch_size = parameters.get('batch_size', 10)
    target_column = parameters.get('target_column', 'Close')
    
    # Create dataset and loader
    test_stock_dataset = StockDataset(test_dataset, ticker='AMZN', target_column=target_column)
    test_loader = DataLoader(test_stock_dataset, batch_size=batch_size, shuffle=False)
    sequence_length = test_stock_dataset.X.shape[1]
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        sequence_length=sequence_length
    )
    
    # Calculate metrics
    test_mse = mean_squared_error(actuals, predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(actuals, predictions)
    test_r2 = r2_score(actuals, predictions)
    
    metrics = {
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'model_type': 'kalman_original',
        'evaluation_data': 'original'
    }
    
    # Create plots
    actual_vs_predicted_plot = _create_actual_vs_predicted_plot(
        actuals, predictions, title='Kalman Model (Original): Actual vs Predicted Stock Prices'
    )
    error_distribution_plot = _create_error_distribution_plot(
        errors, title='Kalman Model (Original): Distribution of Prediction Errors'
    )
    
    log.info(f"Kalman model (original) evaluation completed. Test RMSE: ${metrics['test_rmse']:.2f}, Test R2: {metrics['test_r2']:.4f}")
    
    return metrics, actual_vs_predicted_plot, error_distribution_plot


