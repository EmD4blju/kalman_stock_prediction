# Using Kalman Filters to Improve Stock Price Prediction with LSTM Networks

## 📖 Overview

This project is the practical implementation for the engineering thesis titled _"The use of Kalman filters to improve stock price prediction."_

The primary goal is to investigate whether applying a Kalman filter to a financial time series can enhance the predictive accuracy of a Long Short-Term Memory (LSTM) neural network. The study focuses on predicting the closing price of Amazon (AMZN) stock.

To achieve this, three distinct models are developed and compared:

1.  **Baseline Model**: An LSTM network trained on the original, unprocessed stock price data (OHLC).
2.  **Enhanced Model**: The same LSTM architecture trained on data enriched with technical analysis indicators (RSI, Bollinger Bands).
3.  **Kalman-Filtered Model**: The LSTM network trained on stock price data that has been smoothed using a Kalman filter.

The performance of each model is evaluated and compared to determine the impact of data preprocessing techniques on prediction accuracy.

### Hypothesis

The core hypothesis is that preprocessing the time series data with a Kalman filter will "de-noise" the input signal, leading to a more stable training process and a measurable improvement in the LSTM model's predictive performance compared to the baseline and enhanced models.

## 🛠️ Tech Stack

- **Python 3.12**
- **PyTorch**: For building and training the LSTM models.
- **scikit-learn**: For data normalization (`MinMaxScaler`).
- **Pandas**: For data manipulation and analysis.
- **yfinance**: For downloading historical stock data.
- **Optuna**: For hyperparameter optimization.
- **Matplotlib**: For generating plots and visualizations.

## 📂 Project Structure

```
.
├── documents/              # Thesis-related documents (LaTeX, plans)
├── logs/                   # Log files
├── src/
│   ├── app/                # Main application source
│   │   ├── base_model.ipynb      # Notebook for the baseline model
│   │   ├── enhanced_model.ipynb  # Notebook for the model with technical indicators
│   │   ├── prepare_enhanced.py # Script to prepare data with indicators
│   │   ├── prepare_standard.py # Script to prepare baseline data
│   │   ├── core/             # Core components (data loading, preparation)
│   │   ├── models/           # LSTM model definition
│   │   └── repo/             # Datasets (CSV files)
│   └── tests/                # Unit tests
├── pyproject.toml          # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv)

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/EmD4blju/kalman_stock_prediction.git
    cd kalman_stock_prediction
    ```

2.  Create a virtual environment and install dependencies:
    ```bash
    uv sync
    ```

### Running the Models

The core logic and experiments are contained within the Jupyter Notebooks in `src/app/`. You can run the notebooks to see the data preparation, model training, and evaluation process for each of the three approaches.

1.  **Baseline Model**: Open and run `src/app/base_model.ipynb`.
2.  **Enhanced Model**: Open and run `src/app/enhanced_model.ipynb`.
3.  **Kalman-Filtered Model**: (Notebook to be created)

## 📈 Results

The project systematically evaluates each model based on metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The final comparison, presented in the thesis, will determine which data preparation strategy yields the best results for stock price prediction with LSTMs.
