import torch
from torch import nn
from uuid import uuid4
from logging import getLogger
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import optuna
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
       

log = getLogger('model')

class LSTMStockModel(nn.Module):
    def __init__(self, id:str = uuid4(), ticker:str = 'AMZN', input_dim:int = 1, hidden_dim:int = 5, layer_dim:int = 1, output_dim:int = 1):
        super().__init__()
        self.id = id
        self.ticker = ticker,
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.lstm_layer = nn.LSTM(
            input_dim, 
            hidden_dim, 
            layer_dim, 
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        log.info(self)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            batch_size = x.size(0)
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim)
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim)
        out, (hn, cn) = self.lstm_layer(x, (h0, c0))
        out = self.output_layer(out[:, -1, :])
        return out, (hn, cn)
    
    def __str__(self):
        return (
        f'Model(id={self.id},\n'
        f'ticker={self.ticker},\n'
        f'input_dimension={self.input_dim}),\n'
        f'hidden_dimension={self.hidden_dim},\n'
        f'layer_dimension={self.layer_dim},\n'
        f'output_dimension={self.output_dim})'
    )
    
    def perform_training(self, train_loader, validation_loader, optimizer, loss_function, epochs, feature_number, verbose=True):
        train_mse_list, val_mse_list = [], []
        train_r2_list, val_r2_list = [], []

        for epoch in range(epochs):
            self.train()
            train_actuals, train_predictions = [], []
            for X, y in train_loader:
                optimizer.zero_grad()
                X, y = X.reshape(-1, feature_number, self.input_dim), y.reshape(-1, 1)
                outputs, _ = self(X)
                loss = loss_function(outputs, y)
                loss.backward()
                optimizer.step()
                train_actuals.extend(y.numpy().flatten())
                train_predictions.extend(outputs.detach().numpy().flatten())

            self.eval()
            val_actuals, val_predictions = [], []
            with torch.no_grad():
                for X, y in validation_loader:
                    X, y = X.reshape(-1, feature_number, self.input_dim), y.reshape(-1, 1)
                    outputs, _ = self(X)
                    val_actuals.extend(y.numpy().flatten())
                    val_predictions.extend(outputs.numpy().flatten())

            train_mse = mean_squared_error(train_actuals, train_predictions)
            val_mse = mean_squared_error(val_actuals, val_predictions)
            train_r2 = r2_score(train_actuals, train_predictions)
            val_r2 = r2_score(val_actuals, val_predictions)

            train_mse_list.append(train_mse)
            val_mse_list.append(val_mse)
            train_r2_list.append(train_r2)
            val_r2_list.append(val_r2)

            if verbose:
                log.info(f'Epoch [{epoch+1}/{epochs}]')
                log.info(f'Train MSE: {train_mse:.6f}, R2: {train_r2:.4f}')
                log.info(f'Validation MSE: {val_mse:.6f}, R2: {val_r2:.4f}')

        val_errors = np.array(val_actuals) - np.array(val_predictions)
        return train_mse_list, val_mse_list, train_r2_list, val_r2_list, val_actuals, val_predictions, val_errors

    def evaluate(self, test_loader, loss_function, scaler_y, feature_number):
        self.eval()
        test_loss = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.reshape(-1, feature_number, self.input_dim), y.reshape(-1, 1)
                outputs, _ = self(X)
                test_loss += loss_function(outputs, y).item()
                predictions.extend(outputs.numpy().flatten())
                actuals.extend(y.numpy().flatten())

        test_rmse = np.sqrt(test_loss / len(test_loader))
        print(f'Test RMSE: {test_rmse:.4f}')

        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        
        original_test_loss = np.sqrt(loss_function(torch.tensor(predictions), torch.tensor(actuals)).item())
        print(f'Test Loss (RMSE) on original scale: {original_test_loss:.4f}')

        errors = actuals - predictions
        
        return actuals, predictions, errors
    
    @staticmethod
    def optimize_hyperparameters(train_dataset, validation_dataset, feature_number, ticker, input_dim, output_dim, n_trials=100):

        def objective(trial):
            hidden_dim = trial.suggest_int('hidden_dim', 60, 80)
            layer_dim = trial.suggest_int('layer_dim', 2, 3)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.0015)
            epochs = trial.suggest_int('epochs', 40, 60)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
            validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
            
            model = LSTMStockModel(
                id='temp_opt_model',
                ticker=ticker,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=output_dim
            )
            
            optimizer = Adam(model.parameters(), lr=learning_rate)
            loss_function = MSELoss()
            
            _, val_mse_list, _, _, _, _, _ = model.perform_training(
                train_loader=train_loader,
                validation_loader=validation_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                epochs=epochs,
                feature_number=feature_number,
                verbose=False
            )
            return val_mse_list[-1]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f'Best hyperparameters: {study.best_params}, Best MSE Loss: {study.best_value}')
        return study.best_params, study.best_value