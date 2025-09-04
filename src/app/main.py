import torch
import yfinance
import numpy
import pandas
import matplotlib
import filterpy
from core.data_repository import DataRepository
from core.data_preparator import DataPreparator
from core.dataset import StockDataset
from tools.log_controller import LogController
from torch.utils.data import DataLoader
from logging import getLogger

log_controller = LogController()
log_controller.start()
log = getLogger('main')


log.debug(f'PyTorch: {torch.__version__}')
log.debug(f'YFinance: {yfinance.__version__}')
log.debug(f'Numpy: {numpy.__version__}')
log.debug(f'Pandas: {pandas.__version__}')
log.debug(f'MatPlotLib: {matplotlib.__version__}')
log.debug(f'FilterPy: {filterpy.__version__}')



data_repository = DataRepository()
dataframe = data_repository.get_dataframes()['AMZN']
supervised_data = DataPreparator.reformat_periodic_to_supervised_data(dataframe, k=7)


train_dataset = StockDataset(supervised_data)
train_data_loader = DataLoader(train_dataset, batch_size=20, shuffle=False)

for X,y in train_data_loader:
    print(X, y)
    break