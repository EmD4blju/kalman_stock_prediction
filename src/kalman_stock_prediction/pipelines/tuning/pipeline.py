from kedro.pipeline import Pipeline, node
from .nodes import tune_base_model, tune_enriched_model, tune_kalman_model


def create_tuning_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for hyperparameter optimization using Optuna.
    
    Returns:
        Pipeline with tuning nodes for all three models.
    """
    return Pipeline(
        [
            # Tune base model
            node(
                func=tune_base_model,
                inputs=[
                    "train_dataset",
                    "val_dataset",
                    "scaler_y",
                    "params:tuning"
                ],
                outputs="base_model_best_params",
                name="tune_base_model_node",
            ),
            # Tune enriched model
            node(
                func=tune_enriched_model,
                inputs=[
                    "enriched_train_dataset",
                    "enriched_val_dataset",
                    "enriched_scaler_y",
                    "params:tuning"
                ],
                outputs="enriched_model_best_params",
                name="tune_enriched_model_node",
            ),
            # Tune Kalman model
            node(
                func=tune_kalman_model,
                inputs=[
                    "kalman_train_dataset",
                    "kalman_val_dataset",
                    "kalman_scaler_y",
                    "params:tuning"
                ],
                outputs="kalman_model_best_params",
                name="tune_kalman_model_node",
            ),
        ]
    )
