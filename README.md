# Stock Price Prediction Using Kalman Filter and LSTM Network

This project was developed as part of an engineering thesis at Polish-Japanese Academy of Information-Technology. Its goal is to analyze and predict stock prices using advanced modeling techniques. The core concept involves using the Kalman Filter for initial time-series data denoising and state estimation, followed by a Long Short-Term Memory (LSTM) neural network to forecast future values.

## How to Run

### Prerequisites

- Python 3.12
- `astral-uv` or `pip`

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/EmD4blju/kalman_stock_prediction.git
    cd kalman_stock_prediction
    ```

2.  Install the dependencies:

    There are two recommended ways to install the dependencies for this project: using `uv` or `pip`.

    #### Using `uv` (recommended)

    If you have `uv` from Astral installed, you can sync the environment with a single command:

    ```bash
    uv sync
    ```

    This will create a virtual environment and install all the necessary packages from `pyproject.toml`.

    #### Using `pip`

    If you prefer to use `pip`, it is recommended to first create a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

    Then, install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Kedro Pipeline

To run the full pipeline (processing, training, and evaluation):

```bash
kedro run
```

You can also run individual pipelines by using their names:

```bash
kedro run --pipeline=preprocessing
kedro run --pipeline=modelling
```

### Running the Streamlit Application

To visualize the results, run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Repository Structure

```
kalman_stock_prediction/
│
├── app/
│   └── streamlit_app.py
│
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   └── parameters.yml
│   └── local/
│
├── data/
│   └── raw/
│
├── documents/
│   ├── article/
│   │   └── main.tex
│   └── notes/
│
├── models/
│   ├── base_model/
│   ├── enhanced_model/
│   └── kalman_model/
│
├── notebooks/
│   ├── base_model.ipynb
│   ├── enhanced_model.ipynb
│   └── kalman_model.ipynb
│
├── src/
│   └── kalman_stock_prediction/
│       ├── pipelines/
│       │   ├── preprocessing/
│       │   ├── modelling/
│       │   ├── evaluation/
│       │   └── tuning/
│       ├── models/
│       └── pipeline_registry.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Key Elements

- 📄 **[Scientific Article](documents/article/)**
- 🤖 **[Resulting Models](models/)**
- 📊 **[Datasets](data/)**
- 🛠️ **Kedro Pipelines:**
  - **[Data Preprocessing](src/kalman_stock_prediction/pipelines/preprocessing/)**
  - **[Modeling](src/kalman_stock_prediction/pipelines/modelling/)**
  - **[Model Evaluation](src/kalman_stock_prediction/pipelines/evaluation/)**
  - **[Hyperparameter Tuning](src/kalman_stock_prediction/pipelines/tuning/)**
- 💻 **[Streamlit App](app/streamlit_app.py)**
