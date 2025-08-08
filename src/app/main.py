import torch
import yfinance
import numpy
import pandas
import matplotlib
import filterpy
from core.stock_data_repository import DataRepository


print(
    f'* PyTorch: {torch.__version__}', 
    f'* YFinance: {yfinance.__version__}', 
    f'* Numpy: {numpy.__version__}', 
    f'* Pandas: {pandas.__version__}', 
    f'* MatPlotLib: {matplotlib.__version__}', 
    f'* FilterPy: {filterpy.__version__}',
    sep='\n'
)

data_repository = DataRepository()
data_repository.fetch_periodic_data()
data_repository.save_periodic_data()
print(data_repository.get_dataframes())


