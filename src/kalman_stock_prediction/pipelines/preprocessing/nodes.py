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
    Keeps only Close price and technical indicators, removes OHLCV columns.
    Drops rows with NaN values that result from window-based calculations.
    """
    enriched_dataframe = dataframe.copy()
    
    # Drop 'Date' column if it exists
    if 'Date' in enriched_dataframe.columns:
        enriched_dataframe = enriched_dataframe.drop(columns=['Date'])
    
    # Add RSI (Relative Strength Index)
    enriched_dataframe['RSI'] = RSIIndicator(close=enriched_dataframe[target_column], window=rsi_window).rsi()
    
    # Add Bollinger Bands indicators
    indicator_bb = BollingerBands(close=enriched_dataframe[target_column], window=bb_window, window_dev=bb_window_dev)
    enriched_dataframe['Bandwidth'] = indicator_bb.bollinger_wband()
    enriched_dataframe['%B'] = indicator_bb.bollinger_pband()
    
    # Keep only Close and technical indicators (4 features total)
    columns_to_keep = [target_column, 'RSI', 'Bandwidth', '%B']
    enriched_dataframe = enriched_dataframe[columns_to_keep]
    
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

def reformat_to_supervised(df: pd.DataFrame, target_column: str, timesteps: int) -> pd.DataFrame:
    """
    Reformats a time-series DataFrame into a supervised learning format.
    Uses only the target column (Close) for the base model.

    Args:
        df: The input DataFrame with a time-series index.
        target_column: The name of the column to be predicted.
        timesteps: The number of previous time steps to use as input variables.

    Returns:
        A DataFrame in a supervised learning format.
    """
    data = []
    target_data = df[target_column].values
    
    for i in range(len(target_data) - timesteps):
        # Input sequence (t-timesteps, ..., t-1) - only target column
        input_seq = target_data[i:i + timesteps]
        # Target value (t)
        target_val = target_data[i + timesteps]
        
        # Append target value to input sequence
        row = np.append(input_seq, target_val)
        data.append(row)

    # Create column names
    columns = []
    for t in range(timesteps, 0, -1):
        columns.append(f'{target_column}_t-{t}')
    columns.append(f'{target_column}')

    supervised_df = pd.DataFrame(data, columns=columns)
    return supervised_df

def reformat_enriched_to_supervised(df: pd.DataFrame, target_column: str, timesteps: int) -> pd.DataFrame:
    """
    Reformats an enriched time-series DataFrame into a supervised learning format.
    This function is specifically for datasets with additional features.

    Args:
        df: The input DataFrame with a time-series index and enriched features.
        target_column: The name of the column to be predicted.
        timesteps: The number of previous time steps to use as input variables.

    Returns:
        A DataFrame in a supervised learning format.
    """
    data = []
    for i in range(len(df) - timesteps):
        # Input sequence (t-timesteps, ..., t-1)
        input_seq = df.iloc[i:i + timesteps].values
        # Target value (t)
        target_val = df.iloc[i + timesteps][target_column]
        
        # Flatten input sequence and append target value
        row = np.append(input_seq.flatten(), target_val)
        data.append(row)

    # Create column names
    columns = []
    for t in range(timesteps):
        for col in df.columns:
            columns.append(f'{col}_t-{timesteps-t}')
    columns.append(f'{target_column}')

    supervised_df = pd.DataFrame(data, columns=columns)
    return supervised_df

def fit_scalers(df: pd.DataFrame, scaling_method: str, target_column: str):
    """
    Fits scalers to the training data.

    Args:
        df: The training DataFrame.
        scaling_method: The scaling method to use ('minmax' or 'standard').
        target_column: The name of the target column (e.g., 'Close_t').

    Returns:
        A tuple containing the fitted feature scaler and target scaler.
    """
    if scaling_method == "minmax":
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
    elif scaling_method == "standard":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling_method: {scaling_method}")

    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns]
    y = df[[target_column]]

    scaler_X.fit(X)
    scaler_y.fit(y)

    return scaler_X, scaler_y


def apply_scalers(df: pd.DataFrame, scaler_X, scaler_y, target_column: str) -> pd.DataFrame:
    """
    Applies pre-fitted scalers to a dataframe.
    """
    feature_columns = df.columns.tolist()
    feature_columns.remove(target_column)
    X = df[feature_columns]
    y = df[[target_column]]

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    scaled_df = pd.DataFrame(
        data=X_scaled,
        columns=feature_columns,
        index=df.index
    )
    scaled_df[target_column] = y_scaled
    return scaled_df
    
   
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
