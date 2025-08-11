import pandas as pd
import numpy as np

class DataPreparator():
    """_summary_
    """
    
    @staticmethod
    def reformat_periodic_to_supervised_data(dataframe:pd.DataFrame, target_column:str = 'Close', k:int = 2) -> pd.DataFrame:
        """Reformats **periodic stock market** ```DataFrame``` to **supervised learning** ```DataFrame```.

        Args:
            dataframe (pd.DataFrame): A ```DataFrame``` to reformat.
            k (int): Defines how many steps should the algorithm lookback.

        Returns:
            pd.DataFrame: Reformatted ```DataFrame```.
        """
        
        #~ --- Prepare data for reformatting ---
        data_array = dataframe.to_numpy()
        supervised_dataframe = pd.DataFrame(
            columns=['Date', target_column] + [f'{target_column}_{i}' for i in range(k)],
        )
        dates = dataframe.index
        
        #~ --- Reformat ---
        for i in range(k, len(data_array)):
            current_date = dates[i]
            current_item = data_array[i]
            previous_items = data_array[i-k:i][::-1]
            supervised_dataframe.loc[i-k] = [current_date, current_item, *previous_items]
        supervised_dataframe = supervised_dataframe.set_index(keys=['Date'])
        
        #~ --- Return ---
        return supervised_dataframe