"""Stock Price Prediction Application.

A Streamlit application for testing stock market prediction models.
Uses LangGraph for orchestrating the data loading and prediction workflow.

Note: This module requires the kalman_stock_prediction package to be installed
in development mode (pip install -e .) for proper import resolution.
"""

import streamlit as st
from datetime import datetime, timedelta

from agent import run_prediction


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Stock Price Prediction",
        page_icon="📈",
        layout="centered"
    )
    
    # Header
    st.title("📈 Stock Price Prediction")
    st.markdown("""
    This application uses **LSTM-based models** trained on historical stock data to predict 
    future closing prices. Three different models are available:
    
    - **Base Model**: Uses the last 3 days of closing prices
    - **Enriched Model**: Uses the last 3 days plus technical indicators (RSI, Bollinger Bands)
    - **Kalman Model**: Uses Kalman-filtered prices for noise reduction
    
    ---
    """)
    
    # Sidebar with model information
    with st.sidebar:
        st.header("ℹ️ Model Information")
        
        st.subheader("Base Model")
        st.markdown("""
        - **Input**: Last 3 closing prices
        - **Architecture**: LSTM (hidden_dim=27)
        - **Best for**: Simple trend following
        """)
        
        st.subheader("Enriched Model")
        st.markdown("""
        - **Input**: Last 3 prices + RSI(14) + BB(20)
        - **Architecture**: LSTM (hidden_dim=64, 2 layers)
        - **Best for**: Technical analysis integration
        """)
        
        st.subheader("Kalman Model")
        st.markdown("""
        - **Input**: Last 3 Kalman-filtered prices
        - **Architecture**: LSTM (hidden_dim=27)
        - **Best for**: Noise-reduced predictions
        """)
        
        st.markdown("---")
        st.caption("Models trained on AMZN stock data")
    
    # Main content area
    st.header("🎯 Make a Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Date input
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        
        target_date = st.date_input(
            "Select prediction date",
            value=today,
            min_value=datetime(2022, 1, 1).date(),
            max_value=tomorrow,
            help="Select a date in the past or tomorrow for prediction"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    # Run prediction
    if predict_button:
        with st.spinner("Fetching data and running predictions..."):
            result = run_prediction(
                target_date=target_date.strftime("%Y-%m-%d"),
                ticker="AMZN"
            )
        
        # Display results
        if result.get("error"):
            st.error(f"❌ **Error**: {result['error']}")
        else:
            st.success("✅ Predictions completed successfully!")
            
            # Show input data summary
            st.markdown("---")
            st.subheader("📊 Input Data Summary")
            
            # Show data source
            if result.get("data_source"):
                source = result["data_source"]
                if source == "local":
                    st.info("📁 **Data Source**: Local historical data (yfinance unavailable)")
                else:
                    st.info("🌐 **Data Source**: Yahoo Finance (live data)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Recent Closing Prices (Last 3 days):**")
                if result.get("raw_prices") and result.get("dates"):
                    prices = result["raw_prices"]
                    # Note: prices are reversed (most recent first)
                    for i, price in enumerate(prices):
                        day_label = ["Day -1 (Most Recent)", "Day -2", "Day -3"][i]
                        st.markdown(f"- {day_label}: **${price:.2f}**")
            
            with col2:
                st.markdown("**Technical Indicators:**")
                if result.get("enriched_features"):
                    features = result["enriched_features"]
                    st.markdown(f"- RSI (14-day): **{features['RSI']:.2f}**")
                    st.markdown(f"- Bandwidth (20-day): **{features['Bandwidth']:.4f}**")
                    st.markdown(f"- %B (20-day): **{features['%B']:.4f}**")
            
            # Show predictions
            st.markdown("---")
            st.subheader(f"🔮 Predicted Closing Price for {target_date}")
            
            # Create metrics display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result.get("base_model_prediction"):
                    st.metric(
                        label="Base Model",
                        value=f"${result['base_model_prediction']:.2f}"
                    )
                else:
                    st.metric(label="Base Model", value="N/A")
            
            with col2:
                if result.get("enriched_model_prediction"):
                    st.metric(
                        label="Enriched Model",
                        value=f"${result['enriched_model_prediction']:.2f}"
                    )
                else:
                    st.metric(label="Enriched Model", value="N/A")
            
            with col3:
                if result.get("kalman_model_prediction"):
                    st.metric(
                        label="Kalman Model",
                        value=f"${result['kalman_model_prediction']:.2f}"
                    )
                else:
                    st.metric(label="Kalman Model", value="N/A")
            
            # Average prediction
            predictions = [
                result.get("base_model_prediction"),
                result.get("enriched_model_prediction"),
                result.get("kalman_model_prediction")
            ]
            valid_predictions = [p for p in predictions if p is not None]
            
            if valid_predictions:
                avg_prediction = sum(valid_predictions) / len(valid_predictions)
                st.markdown("---")
                st.markdown(f"### 📈 Average Prediction: **${avg_prediction:.2f}**")
            
            # Additional info
            st.markdown("---")
            st.info("""
            **Note**: These predictions are for educational purposes only. 
            Stock market predictions are inherently uncertain and should not be used as 
            financial advice. Always do your own research before making investment decisions.
            """)
    
    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit & LangGraph | Models trained with PyTorch LSTM")


if __name__ == "__main__":
    main()
