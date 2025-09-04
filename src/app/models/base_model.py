import torch
from torch import nn
from uuid import uuid4
from logging import getLogger

log = getLogger('model')

class BaseStockModel(nn.Module):
    def __init__(self, id:str = uuid4(), ticker:str = 'AMZN', input_dim:int = 5, hidden_dim:int = 5, layer_dim:int = 5, output_dim:int = 5):
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
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        log.debug(f'Forward pass with input shape: {x.shape}, h0 shape: {h0.shape}, c0 shape: {c0.shape}')
        out, (hn, cn) = self.lstm_layer(x, (h0, c0))
        log.debug(f'LSTM output: {out}')
        out = self.output_layer(out[:, -1, :])
        return out, (hn, cn)
    
    def __str__(self):
        return (
        f'Model(id={self.id}, '
        f'ticker={self.ticker}, '
        f'input_dimension={self.input_dim})'
    )