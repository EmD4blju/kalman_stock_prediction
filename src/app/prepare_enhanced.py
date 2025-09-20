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
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

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
amzn = data_repository.get_dataframes()['AMZN']

amzn['RSI'] = RSIIndicator(close=amzn['Close'], window=14).rsi()
indicator_bb = BollingerBands(close=amzn['Close'], window=20, window_dev=2)
amzn['Bandwidth'] = indicator_bb.bollinger_wband()
amzn['%B'] = indicator_bb.bollinger_pband()
amzn.dropna(axis=0, inplace=True)

data_repository.save_periodic_data(dataframe=amzn, path='AMZN_enhanced')

print(amzn.head(20))


