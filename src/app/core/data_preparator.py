import pandas as pd
import numpy as np
from logging import getLogger

log = getLogger('data_preparator')

class DataPreparator():
    """_summary_
    """
    
    @staticmethod
    def reformat_periodic_to_supervised_data(dataframe:pd.DataFrame, target_column:str = 'Close', t:int = 2) -> pd.DataFrame:
        """Reformats **periodic stock market** ```DataFrame``` to **supervised learning** ```DataFrame```.

        Args:
            dataframe (pd.DataFrame): A ```DataFrame``` to reformat.
            t (int): Defines how many steps should the algorithm lookback.

        Returns:
            pd.DataFrame: Reformatted ```DataFrame```.
        """
        
        log.info(f'Preparing data for target column: {target_column}, with t={t}')
        
        #~ --- Prepare data for reformatting ---
        data_array = dataframe[target_column].to_numpy()
        supervised_dataframe = pd.DataFrame(
            columns=['Date', target_column] + [f'{target_column}_{i}' for i in range(t)],
        )
        dates = dataframe.index
        
        #~ --- Reformat ---
        for i in range(t, len(data_array)):
            current_date = dates[i]
            current_item = data_array[i]
            previous_items = data_array[i-t:i][::-1]
            supervised_dataframe.loc[i-t] = [current_date, current_item, *previous_items]
        supervised_dataframe = supervised_dataframe.set_index(keys=['Date'])
        
        #~ --- Return ---
        return supervised_dataframe