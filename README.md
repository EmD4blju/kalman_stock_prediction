# Stock Price Prediction Using Kalman Filter and LSTM Network

This project was developed as part of an engineering thesis. Its goal is to analyze and predict stock prices using advanced modeling techniques. The core concept involves using the Kalman Filter for initial time-series data denoising and state estimation, followed by a Long Short-Term Memory (LSTM) neural network to forecast future values.

## Repository Structure

The repository is organized in a modular fashion to facilitate navigation and project management. Below is a description of the key folders:

- **`/app`**: Contains the [Streamlit](https://streamlit.io/) application, which allows for interactive visualization of model prediction results and their comparison with actual data.

- **`/conf`**: Stores configuration files for the [Kedro](https://kedro.org/) pipelines, including data catalog definitions (`catalog.yml`) and parameters (`parameters.yml`).

- **`/data`**: A central location for all data used in the project. It is divided into subfolders:

  - `raw`: Raw, unprocessed data.
  - `processed`: Data after initial processing.
  - `model_input`: Data prepared as input for the models.
  - `scaled`: Scaled datasets.
  - `scalers`: Saved `Scaler` objects used for data normalization.

- **`/documents`**: Project documentation, including:

  - `article`: The LaTeX source code of the scientific paper describing the methodology and results.
  - `notes`: Notes and supporting materials.

- **`/models`**: Saved, trained prediction models along with their evaluation metrics and best hyperparameters.

- **`/notebooks`**: Jupyter notebooks used for exploratory data analysis (EDA), model prototyping, and results visualization.

- **`/src`**: The main source code of the project, organized as a Python package. It contains data processing logic, model definitions, and Kedro pipelines.

## Key Elements

Below are links to the most important parts of the project:

- **[Scientific Article](/documents/article/)**: Folder containing the LaTeX source code for the paper.
- **[Resulting Models](/models/)**: Directory with trained models, ready for use.
- **[Datasets](/data/)**: Access to all data used and generated in the project.
- **Kedro Pipelines**:
  - **[Data Preprocessing](/src/kalman_stock_prediction/pipelines/preprocessing/)**: The pipeline responsible for cleaning, enriching, and preparing the data.
  - **[Modeling](/src/kalman_stock_prediction/pipelines/modelling/)**: The pipeline for training LSTM models.
  - **[Model Evaluation](/src/kalman_stock_prediction/pipelines/evaluation/)**: The pipeline for assessing the quality of model predictions.
  - **[Hyperparameter Tuning](/src/kalman_stock_prediction/pipelines/tuning/)**: The pipeline for optimizing model hyperparameters.

## How to Run

### Prerequisites

- Python 3.8+
- `pip`

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository-address>
    cd kalman_stock_prediction
    ```

2.  Install the dependencies:
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
