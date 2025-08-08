from yfinance import Ticker, download
from pandas import DataFrame, read_csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

class DataRepository():
    """Module that handles tasks like loading and saving stock market data.
    """
    
    def __init__(self, repo_path:Path = Path('src','app','repo')):
        self._dataframes = {}
        self._repo_path = repo_path
        self._load_all_periodic_data()
    
    def get_dataframes(self) -> Optional[DataFrame]:
        return self._dataframes
    
    def clear_dataframe(self) -> None:
        self._dataframes = {}
    
    def fetch_periodic_data(self, ticker:str = 'AMZN', start:datetime = datetime(2002,10,29), end:datetime = datetime(2003,10,29), interval:str = '1d') -> None:
        """Fetches **singular** periodic stock market data using ```yFinance API``` to a ```DataFrame```. Use ```get_dataframe()``` to access.

        Args:
            tickers (List[str], optional): Stock tickers to download data about. Defaults to ```['AMZN']```
            start (datetime, optional): Start date. Defaults to ```datetime(2002,10,29)```.
            end (datetime, optional): End date. Defaults to ```datetime(2003,10,29)```.
            interval (str, optional): Interval between records. Defaults to ```'1d'```.
        """
        if not ticker in self._dataframes.keys():
            self._dataframes[ticker] = download(
                tickers=[ticker],
                start=start,
                end=end,
                interval=interval,
                multi_level_index=False
            )
        else:
            print(f'{ticker} is already fetched.')
        
    def _load_all_periodic_data(self) -> None:
        """Loads all periodic ```DataFrames``` from configured repository.
        """
        files = self._repo_path.glob(pattern='*.csv')
        for file in files:
            ticker = file.name.removesuffix('.csv')
            path = Path(self._repo_path, file.name)
            self._dataframes[ticker] = read_csv(
                filepath_or_buffer=path,
                sep=';',
                index_col='Date'
            )

        
    def save_periodic_data(self, ticker:str = 'AMZN') -> None:
        """Saves periodic ```DataFrame``` to ```.csv```.

        Args:
            ticker (str, optional): A ticker-specific data to save. Defaults to ```'AMZN'```.
        """
        if ticker in self._dataframes.keys():
            path = Path(self._repo_path, f'{ticker}.csv')
            self._dataframes[ticker].to_csv(
                path_or_buf=path,
                sep=';'
            )
        else:
            print(f'{ticker} does not exist in memory, try to fetch it first.')