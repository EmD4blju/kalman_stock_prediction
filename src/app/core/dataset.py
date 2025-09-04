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
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        log.info(f'Created stock dataset for {ticker}, focusing target: {target_column}')
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Tuple[torch.types.Tensor, torch.types.Tensor]:
        return tensor(self.X.iloc[i].values), tensor(self.y.iloc[i])