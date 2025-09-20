from yfinance import download
from pandas import DataFrame, read_csv
from datetime import datetime
from pathlib import Path
from typing import Optional
from logging import getLogger

log = getLogger('data_repository')

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
    
    def fetch_periodic_data(self, ticker:str = 'AMZN', overwrite:bool = False, start:datetime = datetime(2020,10,29), end:datetime = datetime(2024,10,29), interval:str = '1d') -> None:
        """Fetches **singular** periodic stock market data using ```yFinance API``` to a ```DataFrame```. Use ```get_dataframe()``` to access.

        Args:
            tickers (List[str], optional): Stock tickers to download data about. Defaults to ```['AMZN']```
            start (datetime, optional): Start date. Defaults to ```datetime(2002,10,29)```.
            end (datetime, optional): End date. Defaults to ```datetime(2003,10,29)```.
            interval (str, optional): Interval between records. Defaults to ```'1d'```.
        """
        if overwrite or not ticker in self._dataframes.keys():
            self._dataframes[ticker] = download(
                tickers=[ticker],
                start=start,
                end=end,
                interval=interval,
                multi_level_index=False
            )
        else:
            log.warning(f'{ticker} is already fetched.')
        
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
            log.info(f'Loaded: {ticker}')
        log.info(f'Loaded periodic stock market data for: {set(self._dataframes.keys())}')

    @staticmethod
    def save_periodic_data(dataframe:DataFrame, path:str = 'AMZN') -> None:
        """Saves periodic ```DataFrame``` to ```.csv```.

        Args:
            ticker (str, optional): A ticker-specific data to save. Defaults to ```'AMZN'```.
        """
        dataframe.to_csv(
            path_or_buf=Path('src','app','repo',f'{path}.csv'),
            sep=';',
            index_label='Date'
        )
        log.info(f'Saved: {path}')