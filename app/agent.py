"""LangGraph Agent for stock prediction workflow.

This module implements a LangGraph agent that orchestrates the 
data loading and model prediction workflow for stock price forecasting.

Note: This module requires the kalman_stock_prediction package to be installed
in development mode (pip install -e .) for proper import resolution.
"""

from typing import TypedDict
from datetime import datetime, timedelta
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import torch
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from filterpy.kalman import KalmanFilter
from langgraph.graph import StateGraph, START, END

from kalman_stock_prediction.models.lstm_model import LSTMStockModel


# Project root path
PROJECT_ROOT = Path(__file__).parent.parent


class PredictionState(TypedDict):
    """State for the prediction workflow."""
    # Input
    target_date: str
    ticker: str
    
    # Data loading state
    raw_prices: list[float] | None
    dates: list[str] | None
    enriched_features: dict | None
    data_source: str | None  # 'yfinance' or 'local'
    
    # Prediction outputs (original scale)
    base_model_prediction: float | None
    enriched_model_prediction: float | None
    kalman_model_prediction: float | None
    
    # Error handling
    error: str | None
    status: str


def load_local_data(target_date: datetime, ticker: str = "AMZN") -> pd.DataFrame | None:
    """Load data from local CSV file as fallback.
    
    Args:
        target_date: Target date for prediction
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with historical data or None if not available
    """
    csv_path = PROJECT_ROOT / "data" / "raw" / f"{ticker}.csv"
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path, sep=";", parse_dates=["Date"], index_col="Date")
    return df


def load_stock_data(state: PredictionState) -> PredictionState:
    """Load stock data from yfinance for the target date.
    
    For prediction, we need:
    - Base model: last 3 closing prices before target date
    - Enriched model: last 3 closing prices + RSI(14) + Bandwidth(20) + %B(20)
      This requires at least 20 days of data for Bollinger Bands calculation
    - Kalman model: last 3 closing prices (original, unfiltered)
    
    Falls back to local CSV data if yfinance is unavailable.
    """
    try:
        target_date = datetime.strptime(state["target_date"], "%Y-%m-%d")
        ticker = state.get("ticker", "AMZN")
        
        # We need at least 25 trading days before target date for:
        # - 3 days for the model input
        # - 20 days for Bollinger Bands
        # - Extra buffer for non-trading days
        start_date = target_date - timedelta(days=50)
        end_date = target_date + timedelta(days=1)  # Include target date if available
        
        df = None
        data_source = "yfinance"
        
        # Try to fetch data from yfinance first
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date.strftime("%Y-%m-%d"), 
                             end=end_date.strftime("%Y-%m-%d"))
            if df.empty:
                df = None
        except (ConnectionError, TimeoutError, OSError) as e:
            # Network-related errors - fall back to local data
            df = None
        
        # Fallback to local data if yfinance fails
        if df is None:
            df = load_local_data(target_date, ticker)
            data_source = "local"
            if df is not None:
                # Filter to date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df is None or df.empty:
            return {**state, "error": f"No data available for {ticker}", "status": "error"}
        
        # Convert index to date strings for comparison
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index)
        
        # Check if target date is in the future relative to available data
        latest_available_date = df.index.max()
        target_date_only = pd.Timestamp(target_date.date())
        
        if target_date_only > latest_available_date:
            # Target is in the future - use all available data up to latest
            df_filtered = df
        else:
            # Target is in the past - filter to data before and including target
            df_filtered = df[df.index <= target_date_only]
        
        if len(df_filtered) < 23:  # Need at least 20 for BB + 3 for model input
            return {**state, "error": f"Insufficient data. Need at least 23 days, got {len(df_filtered)}", "status": "error"}
        
        # Get close prices
        close_prices = df_filtered["Close"].values
        dates = [d.strftime("%Y-%m-%d") for d in df_filtered.index]
        
        # Last 3 prices for base model (most recent first as per preprocessing)
        raw_prices = close_prices[-3:][::-1].tolist()  # Reverse to match model expectation
        
        # Calculate enriched features using the full data
        # RSI with 14-day window
        rsi_indicator = RSIIndicator(close=df_filtered["Close"], window=14)
        rsi_values = rsi_indicator.rsi()
        
        # Bollinger Bands with 20-day window
        bb_indicator = BollingerBands(close=df_filtered["Close"], window=20, window_dev=2)
        bandwidth_values = bb_indicator.bollinger_wband()
        percent_b_values = bb_indicator.bollinger_pband()
        
        # Get the latest enriched features (for the prediction date)
        enriched_features = {
            "RSI": float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else 50.0,
            "Bandwidth": float(bandwidth_values.iloc[-1]) if not pd.isna(bandwidth_values.iloc[-1]) else 0.0,
            "%B": float(percent_b_values.iloc[-1]) if not pd.isna(percent_b_values.iloc[-1]) else 0.5,
        }
        
        return {
            **state,
            "raw_prices": raw_prices,
            "dates": dates[-3:],
            "enriched_features": enriched_features,
            "data_source": data_source,
            "status": "data_loaded"
        }
        
    except Exception as e:
        return {**state, "error": str(e), "status": "error"}


def predict_base_model(state: PredictionState) -> PredictionState:
    """Make prediction using the base model."""
    if state.get("error"):
        return state
    
    try:
        # Load model (weights_only=False is required because we store model config
        # alongside weights - trusted model files from this repository)
        model_path = PROJECT_ROOT / "models" / "base_model" / "model.pth"
        model_data = torch.load(model_path, weights_only=False)
        config = model_data['model_config']
        
        model = LSTMStockModel(
            id=config['id'],
            ticker=config['ticker'],
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            layer_dim=config['layer_dim'],
            output_dim=config['output_dim']
        )
        model.load_state_dict(model_data['state_dict'])
        model.eval()
        
        # Load scalers
        scaler_X_path = PROJECT_ROOT / "data" / "scalers" / "scaler_X.pkl"
        scaler_y_path = PROJECT_ROOT / "data" / "scalers" / "scaler_y.pkl"
        
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        # Prepare input - base model uses 3 lagged prices as features
        raw_prices = np.array(state["raw_prices"]).reshape(1, -1)
        scaled_input = scaler_X.transform(raw_prices)
        
        # Reshape for LSTM: (batch, seq_len, features)
        feature_number = scaled_input.shape[1]  # 3 features
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        input_tensor = input_tensor.reshape(-1, feature_number, 1)
        
        # Predict
        with torch.no_grad():
            output, _ = model(input_tensor)
            scaled_prediction = output.numpy().flatten()[0]
        
        # Inverse transform to original scale
        prediction = scaler_y.inverse_transform([[scaled_prediction]])[0][0]
        
        return {**state, "base_model_prediction": float(prediction)}
        
    except Exception as e:
        return {**state, "error": f"Base model error: {str(e)}", "status": "error"}


def predict_enriched_model(state: PredictionState) -> PredictionState:
    """Make prediction using the enriched model."""
    if state.get("error"):
        return state
    
    try:
        # Load model (weights_only=False is required because we store model config
        # alongside weights - trusted model files from this repository)
        model_path = PROJECT_ROOT / "models" / "enhanced_model" / "model.pth"
        model_data = torch.load(model_path, weights_only=False)
        config = model_data['model_config']
        
        model = LSTMStockModel(
            id=config['id'],
            ticker=config['ticker'],
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            layer_dim=config['layer_dim'],
            output_dim=config['output_dim']
        )
        model.load_state_dict(model_data['state_dict'])
        model.eval()
        
        # Load scalers
        scaler_X_path = PROJECT_ROOT / "data" / "scalers" / "enriched_scaler_X.pkl"
        scaler_y_path = PROJECT_ROOT / "data" / "scalers" / "enriched_scaler_y.pkl"
        
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        # Prepare input - enriched model uses 3 lagged prices + RSI + Bandwidth + %B
        # Feature order: Close_0, Close_1, Close_2, RSI, Bandwidth, %B
        raw_prices = state["raw_prices"]
        enriched_features = state["enriched_features"]
        
        features = raw_prices + [
            enriched_features["RSI"],
            enriched_features["Bandwidth"],
            enriched_features["%B"]
        ]
        
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler_X.transform(input_array)
        
        # Reshape for LSTM: (batch, seq_len, features)
        feature_number = scaled_input.shape[1]  # 6 features
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        input_tensor = input_tensor.reshape(-1, feature_number, 1)
        
        # Predict
        with torch.no_grad():
            output, _ = model(input_tensor)
            scaled_prediction = output.numpy().flatten()[0]
        
        # Inverse transform to original scale
        prediction = scaler_y.inverse_transform([[scaled_prediction]])[0][0]
        
        return {**state, "enriched_model_prediction": float(prediction)}
        
    except Exception as e:
        return {**state, "error": f"Enriched model error: {str(e)}", "status": "error"}


def predict_kalman_model(state: PredictionState) -> PredictionState:
    """Make prediction using the Kalman model."""
    if state.get("error"):
        return state
    
    try:
        # Load model (weights_only=False is required because we store model config
        # alongside weights - trusted model files from this repository)
        model_path = PROJECT_ROOT / "models" / "kalman_model" / "model.pth"
        model_data = torch.load(model_path, weights_only=False)
        config = model_data['model_config']
        
        model = LSTMStockModel(
            id=config['id'],
            ticker=config['ticker'],
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            layer_dim=config['layer_dim'],
            output_dim=config['output_dim']
        )
        model.load_state_dict(model_data['state_dict'])
        model.eval()
        
        # Load scalers
        scaler_X_path = PROJECT_ROOT / "data" / "scalers" / "kalman_scaler_X.pkl"
        scaler_y_path = PROJECT_ROOT / "data" / "scalers" / "kalman_scaler_y.pkl"
        
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        # Prepare input - kalman model uses 3 lagged prices (original, unfiltered)
        # Same as base model - the model was trained on original data
        raw_prices = np.array(state["raw_prices"]).reshape(1, -1)
        scaled_input = scaler_X.transform(raw_prices)
        
        # Reshape for LSTM: (batch, seq_len, features)
        feature_number = scaled_input.shape[1]  # 3 features
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32)
        input_tensor = input_tensor.reshape(-1, feature_number, 1)
        
        # Predict
        with torch.no_grad():
            output, _ = model(input_tensor)
            scaled_prediction = output.numpy().flatten()[0]
        
        # Inverse transform to original scale
        prediction = scaler_y.inverse_transform([[scaled_prediction]])[0][0]
        
        return {**state, "kalman_model_prediction": float(prediction), "status": "completed"}
        
    except Exception as e:
        return {**state, "error": f"Kalman model error: {str(e)}", "status": "error"}


def build_prediction_graph() -> StateGraph:
    """Build the LangGraph prediction workflow."""
    
    # Create the graph
    workflow = StateGraph(PredictionState)
    
    # Add nodes
    workflow.add_node("load_data", load_stock_data)
    workflow.add_node("predict_base", predict_base_model)
    workflow.add_node("predict_enriched", predict_enriched_model)
    workflow.add_node("predict_kalman", predict_kalman_model)
    
    # Define the flow
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "predict_base")
    workflow.add_edge("predict_base", "predict_enriched")
    workflow.add_edge("predict_enriched", "predict_kalman")
    workflow.add_edge("predict_kalman", END)
    
    return workflow.compile()


def run_prediction(target_date: str, ticker: str = "AMZN") -> dict:
    """Run the complete prediction workflow.
    
    Args:
        target_date: Target date in YYYY-MM-DD format
        ticker: Stock ticker symbol (default: AMZN)
    
    Returns:
        Dictionary with prediction results or error information
    """
    graph = build_prediction_graph()
    
    initial_state: PredictionState = {
        "target_date": target_date,
        "ticker": ticker,
        "raw_prices": None,
        "dates": None,
        "enriched_features": None,
        "data_source": None,
        "base_model_prediction": None,
        "enriched_model_prediction": None,
        "kalman_model_prediction": None,
        "error": None,
        "status": "initialized"
    }
    
    result = graph.invoke(initial_state)
    return result
