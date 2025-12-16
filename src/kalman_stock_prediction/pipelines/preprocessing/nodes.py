import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from filterpy.kalman import KalmanFilter

def enrich_dataset(dataframe: pd.DataFrame, target_column: str, rsi_window: int = 14, bb_window: int = 20, bb_window_dev: int = 2) -> pd.DataFrame:
    """
    Enrich the dataset with technical indicators: RSI, Bandwidth, and %B.
    Drops rows with NaN values that result from window-based calculations.
    """
    enriched_dataframe = dataframe.copy()
    
    # Add RSI (Relative Strength Index)
    enriched_dataframe['RSI'] = RSIIndicator(close=enriched_dataframe[target_column], window=rsi_window).rsi()
    
    # Add Bollinger Bands indicators
    indicator_bb = BollingerBands(close=enriched_dataframe[target_column], window=bb_window, window_dev=bb_window_dev)
    enriched_dataframe['Bandwidth'] = indicator_bb.bollinger_wband()
    enriched_dataframe['%B'] = indicator_bb.bollinger_pband()
    
    # Drop rows with NaN values caused by window calculations
    enriched_dataframe = enriched_dataframe.dropna()
    
    return enriched_dataframe

def apply_kalman_filter(dataframe: pd.DataFrame, target_column: str, kalman_F: list, kalman_H: list, kalman_P: list, kalman_R: float, kalman_Q: list) -> pd.DataFrame:
    """
    Applies Kalman filter to the target column of the passed DataFrame.
    
    Args:
        dataframe: A DataFrame to apply Kalman filter to.
        target_column: The target column to apply Kalman filter to.
        kalman_F: State transition matrix (2x2).
        kalman_H: Measurement function (1x2).
        kalman_P: Covariance matrix initial value (2x2).
        kalman_R: State uncertainty.
        kalman_Q: Process uncertainty (2x2).
    
    Returns:
        A DataFrame with Kalman filter applied to the target column.
    """
    filtered_dataframe = dataframe.copy()
    
    #~ --- Prepare data for filtering ---
    data_array = dataframe[target_column].to_numpy()
    
    #~ --- Create Kalman filter ---
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data_array[0]], [0.]])       # initial state (price and velocity)
    kf.F = np.array(kalman_F)                      # state transition matrix
    kf.H = np.array(kalman_H)                      # measurement function
    kf.P = np.array(kalman_P)               # covariance matrix
    kf.R = kalman_R                                # state uncertainty
    kf.Q = np.array(kalman_Q)                      # process uncertainty
    
    #~ --- Apply Kalman filter ---
    filtered_data = []
    for z in data_array:
        kf.predict()
        kf.update(z)
        filtered_data.append(kf.x[0, 0])
    
    #~ --- Return ---
    filtered_dataframe[target_column] = filtered_data
    return filtered_dataframe

def reformat_periodic_to_supervised_data(dataframe:pd.DataFrame, target_column:str, timesteps:int) -> pd.DataFrame:

    #~ --- Prepare data for reformatting ---
    data_array = dataframe[target_column].to_numpy()
    dates = dataframe.index
    
    #~ --- Check for enrichment columns ---
    enrichment_columns = []
    has_enrichment = {'RSI', 'Bandwidth', '%B'}.issubset(dataframe.columns)
    if has_enrichment:
        enrichment_columns = ['RSI', 'Bandwidth', '%B']
    
    #~ --- Build column names ---
    column_names = ['Date']
    
    if has_enrichment:
        # Create columns for each timestep with all features
        for i in range(timesteps - 1, -1, -1):  # timesteps-1, timesteps-2, ..., 1, 0
            column_names.append(f'{target_column}_{i}')
            for enr_col in enrichment_columns:
                column_names.append(f'{enr_col}_{i}')
    else:
        # Create columns for regular data (only Close values)
        for i in range(timesteps - 1, -1, -1):  # timesteps-1, timesteps-2, ..., 1, 0
            column_names.append(f'{target_column}_{i}')
    
    # Add target column
    column_names.append(target_column)
    
    supervised_dataframe = pd.DataFrame(columns=column_names)
    
    #~ --- Reformat ---
    for i in range(timesteps, len(data_array)):
        current_date = dates[i]
        current_item = data_array[i]
        
        row_data = [current_date]
        
        if has_enrichment:
            # Add historical timesteps with all features
            for j in range(i - timesteps, i):  # From oldest to newest
                row_data.append(data_array[j])
                for enr_col in enrichment_columns:
                    row_data.append(dataframe.loc[dates[j], enr_col])
        else:
            # Add only Close values for historical timesteps
            previous_items = data_array[i-timesteps:i][::-1]
            row_data.extend(previous_items)
        
        # Add target value
        row_data.append(current_item)
        
        supervised_dataframe.loc[i-timesteps] = row_data
    
    supervised_dataframe = supervised_dataframe.set_index(keys=['Date'])
    
    #~ --- Return ---
    return supervised_dataframe


def scale_dataframe(dataframe:pd.DataFrame, scaling_method: str, target_column: str) -> pd.DataFrame:
    
    match scaling_method:
        case "minmax":
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        
        case "standard":
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
        case _:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
        
    
    feature_columns = dataframe.columns.tolist()
    feature_columns.remove(target_column)   
    X = dataframe[feature_columns]
    y = dataframe[[target_column]]
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    scaled_dataframe = pd.DataFrame(
        data = X_scaled,
        columns = feature_columns,
        index = dataframe.index
    )
    scaled_dataframe[target_column] = y_scaled
    return scaled_dataframe, scaler_X, scaler_y
    
   
def split_dataframe(dataframe:pd.DataFrame, val_size:float=0.2, test_size:float=0.1):

    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        shuffle=False
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        shuffle=False
    )
    

    return train_df, val_df, test_df
    