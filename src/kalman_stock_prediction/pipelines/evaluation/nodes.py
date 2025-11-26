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
    feature_number = test_stock_dataset.X.shape[1]
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        feature_number=feature_number
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
    feature_number = test_stock_dataset.X.shape[1]
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        feature_number=feature_number
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
    feature_number = test_stock_dataset.X.shape[1]
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        feature_number=feature_number
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
    feature_number = test_stock_dataset.X.shape[1]
    
    # Setup loss function
    loss_function = MSELoss()
    
    # Evaluate model
    actuals, predictions, errors = model.evaluate(
        test_loader=test_loader,
        loss_function=loss_function,
        scaler_y=scaler_y,
        feature_number=feature_number
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
