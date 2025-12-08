"""Nodes for model training pipeline."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import pandas as pd
from typing import Dict, Any
from logging import getLogger

from ...models.lstm_model import LSTMStockModel
from ...models.dataset import StockDataset

log = getLogger(__name__)


def train_base_model(
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    scaler_y: Any,
    parameters: Dict[str, Any]
) -> tuple[LSTMStockModel, Dict[str, Any]]:
    """
    Train base LSTM model on standard features.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        scaler_y: Scaler for target variable (for inverse transform)
        parameters: Training parameters (epochs, batch_size, learning_rate, hidden_dim, layer_dim)
    
    Returns:
        Trained model and training metrics dictionary
    """
    log.info("Starting base model training")
    
    # Extract parameters
    epochs = parameters.get('epochs', 20)
    batch_size = parameters.get('batch_size', 16)
    learning_rate = parameters.get('learning_rate', 0.001)
    hidden_dim = parameters.get('hidden_dim', 27)
    layer_dim = parameters.get('layer_dim', 1)
    target_column = parameters.get('target_column', 'Close')
    
    # Create datasets
    train_stock_dataset = StockDataset(train_dataset, ticker='AMZN', target_column=target_column)
    val_stock_dataset = StockDataset(val_dataset, ticker='AMZN', target_column=target_column)
    
    # Create data loaders
    train_loader = DataLoader(train_stock_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_stock_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    sequence_length = train_stock_dataset.X.shape[1]
    model = LSTMStockModel(
        id='base_model',
        ticker='AMZN',
        input_dim=1,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=1
    )
    
    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = MSELoss()
    
    # Train model
    train_mse_list, val_mse_list, train_r2_list, val_r2_list, val_actuals, val_predictions, val_errors = model.perform_training(
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=epochs,
        sequence_length=sequence_length
    )
    
    # Calculate RMSE on original scale (human-readable)
    val_actuals_original = scaler_y.inverse_transform(np.array(val_actuals).reshape(-1, 1)).flatten()
    val_predictions_original = scaler_y.inverse_transform(np.array(val_predictions).reshape(-1, 1)).flatten()
    val_rmse_original = np.sqrt(np.mean((val_actuals_original - val_predictions_original) ** 2))
    
    # Compile metrics
    metrics = {
        'final_train_mse': float(train_mse_list[-1]),
        'final_val_mse': float(val_mse_list[-1]),
        'final_train_r2': float(train_r2_list[-1]),
        'final_val_r2': float(val_r2_list[-1]),
        'final_val_rmse_original_scale': float(val_rmse_original),
        'train_mse_history': [float(x) for x in train_mse_list],
        'val_mse_history': [float(x) for x in val_mse_list],
        'train_r2_history': [float(x) for x in train_r2_list],
        'val_r2_history': [float(x) for x in val_r2_list],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'layer_dim': layer_dim,
        'model_type': 'base'
    }
    
    log.info(f"Base model training completed. Final val MSE: {metrics['final_val_mse']:.6f}, Final val R2: {metrics['final_val_r2']:.4f}, Final val RMSE (original scale): ${metrics['final_val_rmse_original_scale']:.2f}")
    
    return model, metrics


def train_enriched_model(
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    scaler_y: Any,
    parameters: Dict[str, Any]
) -> tuple[LSTMStockModel, Dict[str, Any]]:
    """
    Train enhanced LSTM model with enriched features (RSI, Bandwidth, %B).
    
    Args:
        train_dataset: Training dataset with enriched features
        val_dataset: Validation dataset with enriched features
        scaler_y: Scaler for target variable (for inverse transform)
        parameters: Training parameters (epochs, batch_size, learning_rate, hidden_dim, layer_dim)
    
    Returns:
        Trained model and training metrics dictionary
    """
    log.info("Starting enriched model training")
    
    # Extract parameters
    epochs = parameters.get('epochs', 50)
    batch_size = parameters.get('batch_size', 16)
    learning_rate = parameters.get('learning_rate', 0.001)
    hidden_dim = parameters.get('hidden_dim', 64)
    layer_dim = parameters.get('layer_dim', 2)
    target_column = parameters.get('target_column', 'Close')
    
    # Create datasets
    train_stock_dataset = StockDataset(train_dataset, ticker='AMZN_ENHANCED', target_column=target_column)
    val_stock_dataset = StockDataset(val_dataset, ticker='AMZN_ENHANCED', target_column=target_column)
    
    # Create data loaders
    train_loader = DataLoader(train_stock_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_stock_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = 4
    sequence_length = train_stock_dataset.X.shape[1] // input_dim
    
    model = LSTMStockModel(
        id='enriched_model',
        ticker='AMZN_ENHANCED',
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=1
    )
    
    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = MSELoss()
    
    # Train model
    train_mse_list, val_mse_list, train_r2_list, val_r2_list, val_actuals, val_predictions, val_errors = model.perform_training(
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=epochs,
        sequence_length=sequence_length
    )
    
    # Calculate RMSE on original scale
    val_actuals_original = scaler_y.inverse_transform(np.array(val_actuals).reshape(-1, 1)).flatten()
    val_predictions_original = scaler_y.inverse_transform(np.array(val_predictions).reshape(-1, 1)).flatten()
    val_rmse_original = np.sqrt(np.mean((val_actuals_original - val_predictions_original) ** 2))
    
    # Compile metrics
    metrics = {
        'final_train_mse': float(train_mse_list[-1]),
        'final_val_mse': float(val_mse_list[-1]),
        'final_train_r2': float(train_r2_list[-1]),
        'final_val_r2': float(val_r2_list[-1]),
        'final_val_rmse_original_scale': float(val_rmse_original),
        'train_mse_history': [float(x) for x in train_mse_list],
        'val_mse_history': [float(x) for x in val_mse_list],
        'train_r2_history': [float(x) for x in train_r2_list],
        'val_r2_history': [float(x) for x in val_r2_list],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'layer_dim': layer_dim,
        'model_type': 'enriched'
    }
    
    log.info(f"Enriched model training completed. Final val MSE: {metrics['final_val_mse']:.6f}, Final val R2: {metrics['final_val_r2']:.4f}, Final val RMSE (original scale): ${metrics['final_val_rmse_original_scale']:.2f}")
    
    return model, metrics


def train_kalman_model(
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    scaler_y: Any,
    parameters: Dict[str, Any]
) -> tuple[LSTMStockModel, Dict[str, Any]]:
    """
    Train LSTM model on Kalman-filtered data.
    
    Args:
        train_dataset: Training dataset with Kalman-filtered features
        val_dataset: Validation dataset with Kalman-filtered features
        scaler_y: Scaler for target variable (for inverse transform)
        parameters: Training parameters (epochs, batch_size, learning_rate, hidden_dim, layer_dim)
    
    Returns:
        Trained model and training metrics dictionary
    """
    log.info("Starting Kalman model training")
    
    # Extract parameters
    epochs = parameters.get('epochs', 20)
    batch_size = parameters.get('batch_size', 16)
    learning_rate = parameters.get('learning_rate', 0.001)
    hidden_dim = parameters.get('hidden_dim', 27)
    layer_dim = parameters.get('layer_dim', 1)
    target_column = parameters.get('target_column', 'Close')
    
    # Create datasets
    train_stock_dataset = StockDataset(train_dataset, ticker='AMZN_KALMAN', target_column=target_column)
    val_stock_dataset = StockDataset(val_dataset, ticker='AMZN_KALMAN', target_column=target_column)
    
    # Create data loaders
    train_loader = DataLoader(train_stock_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_stock_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    sequence_length = train_stock_dataset.X.shape[1]
    model = LSTMStockModel(
        id='kalman_model',
        ticker='AMZN_KALMAN',
        input_dim=1,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=1
    )
    
    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = MSELoss()
    
    # Train model
    train_mse_list, val_mse_list, train_r2_list, val_r2_list, val_actuals, val_predictions, val_errors = model.perform_training(
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=epochs,
        sequence_length=sequence_length
    )
    
    # Calculate RMSE on original scale (human-readable)
    val_actuals_original = scaler_y.inverse_transform(np.array(val_actuals).reshape(-1, 1)).flatten()
    val_predictions_original = scaler_y.inverse_transform(np.array(val_predictions).reshape(-1, 1)).flatten()
    val_rmse_original = np.sqrt(np.mean((val_actuals_original - val_predictions_original) ** 2))
    
    # Compile metrics
    metrics = {
        'final_train_mse': float(train_mse_list[-1]),
        'final_val_mse': float(val_mse_list[-1]),
        'final_train_r2': float(train_r2_list[-1]),
        'final_val_r2': float(val_r2_list[-1]),
        'final_val_rmse_original_scale': float(val_rmse_original),
        'train_mse_history': [float(x) for x in train_mse_list],
        'val_mse_history': [float(x) for x in val_mse_list],
        'train_r2_history': [float(x) for x in train_r2_list],
        'val_r2_history': [float(x) for x in val_r2_list],
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'layer_dim': layer_dim,
        'model_type': 'kalman'
    }
    
    log.info(f"Kalman model training completed. Final val MSE: {metrics['final_val_mse']:.6f}, Final val R2: {metrics['final_val_r2']:.4f}, Final val RMSE (original scale): ${metrics['final_val_rmse_original_scale']:.2f}")
    
    return model, metrics
