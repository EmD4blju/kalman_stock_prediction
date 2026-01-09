import pandas as pd
import torch
import json
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import optuna
from logging import getLogger
from kalman_stock_prediction.models.dataset import StockDataset
from kalman_stock_prediction.models.lstm_model import LSTMStockModel

log = getLogger(__name__)


def tune_base_model(
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    scaler_y,
    params: dict
) -> dict:
    """
    Optimize hyperparameters for the base model using Optuna.
    
    Args:
        train_dataset: Training data
        val_dataset: Validation data
        scaler_y: Fitted target scaler
        params: Dictionary with tuning parameters
    
    Returns:
        Dictionary with best hyperparameters
    """
    log.info("Starting hyperparameter optimization for base model...")
    
    target_column = params['target_column']
    timesteps = params['timesteps']
    n_trials = params.get('n_trials', 100)
    
    # Prepare datasets
    train_torch = StockDataset(train_dataset, ticker='AMZN', target_column=target_column)
    val_torch = StockDataset(val_dataset, ticker='AMZN', target_column=target_column)
    
    def objective(trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 20, 50)
        layer_dim = trial.suggest_int('layer_dim', 1, 3)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        epochs = trial.suggest_int('epochs', 10, 50)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False)
        
        # For base model: input_dim=1 (only Close), sequence_length=timesteps
        model = LSTMStockModel(
            id='base_tuning',
            ticker='AMZN',
            input_dim=1,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            output_dim=1
        )
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_function = MSELoss()
        
        _, val_mse_list, _, _, _, _, _ = model.perform_training(
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=epochs,
            sequence_length=timesteps,
            verbose=False
        )
        
        return val_mse_list[-1]
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['best_val_mse'] = study.best_value
    
    log.info(f"Best hyperparameters for base model: {best_params}")
    log.info(f"Best validation MSE: {study.best_value:.6f}")
    
    return best_params


def tune_enriched_model(
    enriched_train_dataset: pd.DataFrame,
    enriched_val_dataset: pd.DataFrame,
    enriched_scaler_y,
    params: dict
) -> dict:
    """
    Optimize hyperparameters for the enriched model using Optuna.
    
    Args:
        enriched_train_dataset: Training data with technical indicators
        enriched_val_dataset: Validation data with technical indicators
        enriched_scaler_y: Fitted target scaler
        params: Dictionary with tuning parameters
    
    Returns:
        Dictionary with best hyperparameters
    """
    log.info("Starting hyperparameter optimization for enriched model...")
    
    target_column = params['target_column']
    timesteps = params['timesteps']
    n_trials = params.get('n_trials', 100)
    
    
    # Prepare datasets
    train_torch = StockDataset(enriched_train_dataset, ticker='AMZN', target_column=target_column)
    val_torch = StockDataset(enriched_val_dataset, ticker='AMZN', target_column=target_column)
    
    def objective(trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 50, 100)
        layer_dim = trial.suggest_int('layer_dim', 1, 3)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        epochs = trial.suggest_int('epochs', 20, 80)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False)
        
        model = LSTMStockModel(
            id='enriched_tuning',
            ticker='AMZN',
            input_dim=4,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            output_dim=1
        )
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_function = MSELoss()
        
        _, val_mse_list, _, _, _, _, _ = model.perform_training(
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=epochs,
            sequence_length=timesteps,
            verbose=False
        )
        
        return val_mse_list[-1]
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['best_val_mse'] = study.best_value
    best_params['input_dim'] = 4
    
    log.info(f"Best hyperparameters for enriched model: {best_params}")
    log.info(f"Best validation MSE: {study.best_value:.6f}")
    
    return best_params


def tune_kalman_model(
    kalman_train_dataset: pd.DataFrame,
    kalman_val_dataset: pd.DataFrame,
    kalman_scaler_y,
    params: dict
) -> dict:
    """
    Optimize hyperparameters for the Kalman filtered model using Optuna.
    
    Args:
        kalman_train_dataset: Training data with Kalman filter applied
        kalman_val_dataset: Validation data with Kalman filter applied
        kalman_scaler_y: Fitted target scaler
        params: Dictionary with tuning parameters
    
    Returns:
        Dictionary with best hyperparameters
    """
    log.info("Starting hyperparameter optimization for Kalman model...")
    
    target_column = params['target_column']
    timesteps = params['timesteps']
    n_trials = params.get('n_trials', 100)
    
    # Prepare datasets
    train_torch = StockDataset(kalman_train_dataset, ticker='AMZN', target_column=target_column)
    val_torch = StockDataset(kalman_val_dataset, ticker='AMZN', target_column=target_column)
    
    def objective(trial):
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 20, 50)
        layer_dim = trial.suggest_int('layer_dim', 1, 3)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        epochs = trial.suggest_int('epochs', 10, 50)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False)
        
        # For Kalman model: input_dim=1 (only Close), sequence_length=timesteps
        model = LSTMStockModel(
            id='kalman_tuning',
            ticker='AMZN',
            input_dim=1,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            output_dim=1
        )
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_function = MSELoss()
        
        _, val_mse_list, _, _, _, _, _ = model.perform_training(
            train_loader=train_loader,
            validation_loader=val_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=epochs,
            sequence_length=timesteps,
            verbose=False
        )
        
        return val_mse_list[-1]
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params['best_val_mse'] = study.best_value
    
    log.info(f"Best hyperparameters for Kalman model: {best_params}")
    log.info(f"Best validation MSE: {study.best_value:.6f}")
    
    return best_params
