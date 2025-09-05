import torch
import pandas as pd
from torch.utils.data import Dataset
from torch import tensor
from typing import Tuple
from logging import getLogger

log = getLogger('dataset')

class StockDataset(Dataset):
    def __init__(self, data:pd.DataFrame, ticker:str = 'AMZN', target_column:str = 'Close'):
        self.ticker = ticker
        self.target_column = target_column
        self.data = data
        self.X = tensor(self.data.drop(columns=[target_column]).to_numpy(), dtype=torch.float32)
        self.y = tensor(self.data[target_column].to_numpy(), dtype=torch.float32)
        log.info(f'Created stock dataset for {ticker}, focusing target: {target_column}')
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Tuple[torch.types.Tensor, torch.types.Tensor]:
        return self.X[i], self.y[i]