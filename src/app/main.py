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
data_repository.fetch_periodic_data(ticker='AMZN', overwrite=True, start=datetime(2018,1,1), end=datetime(2024, 1, 1))
data_repository.save_periodic_data(ticker='AMZN')
