import pytest
from src.app.core.data_preparator import DataPreparator
from pandas import read_csv
from pandas.testing import assert_frame_equal
import numpy as np
from pathlib import Path

def test_reformat_should_handle_close_column():
    
    #~ --- Arrange dataframes ---
    periodic_dataframe = read_csv(
        filepath_or_buffer=Path('src', 'tests', 'repo', 'test_dataframe_in.csv'),
        sep=';',
        index_col=0,
    )['Close']
    
    supervised_dataframe = read_csv(
        filepath_or_buffer=Path('src', 'tests', 'repo', 'test_dataframe_out_close.csv'),
        sep=';',
        index_col=0,
    )
    
    #~ --- Perform action ---
    result_dataframe = DataPreparator.reformat_periodic_to_supervised_data(periodic_dataframe)
    
    #~ --- Make assertion ---
    assert_frame_equal(supervised_dataframe, result_dataframe)

def test_reformat_should_handle_open_column():
    #~ --- Arrange dataframes ---
    periodic_dataframe = read_csv(
        filepath_or_buffer=Path('src', 'tests', 'repo', 'test_dataframe_in.csv'),
        sep=';',
        index_col=0,
    )['Open']
    
    supervised_dataframe = read_csv(
        filepath_or_buffer=Path('src', 'tests', 'repo', 'test_dataframe_out_open.csv'),
        sep=';',
        index_col=0,
    )
    
    #~ --- Perform action ---
    result_dataframe = DataPreparator.reformat_periodic_to_supervised_data(periodic_dataframe, target_column='Open')
    
    
    #~ --- Make assertion ---
    assert_frame_equal(supervised_dataframe, result_dataframe)