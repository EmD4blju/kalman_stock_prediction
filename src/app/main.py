import torch
import yfinance
import numpy
import pandas
import matplotlib
import filterpy


print(
    f'* PyTorch: {torch.__version__}', 
    f'* YFinance: {yfinance.__version__}', 
    f'* Numpy: {numpy.__version__}', 
    f'* Pandas: {pandas.__version__}', 
    f'* MatPlotLib: {matplotlib.__version__}', 
    f'* FilterPy: {filterpy.__version__}',
    sep='\n'
)