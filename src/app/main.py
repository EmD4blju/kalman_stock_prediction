import torch
import yfinance
import numpy
import pandas
import matplotlib
import filterpy
from core.data_repository import DataRepository
from tools.log_controller import LogController
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
data_repository.fetch_periodic_data()
data_repository.save_periodic_data()
log.info(f'Loaded periodic stock market data for: {data_repository.get_dataframes().keys()}')


