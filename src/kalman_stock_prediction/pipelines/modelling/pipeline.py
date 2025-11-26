"""Pipeline for model training."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_base_model, train_enriched_model, train_kalman_model


def create_modelling_pipeline(**kwargs) -> Pipeline:
    """
    Create the modelling pipeline with three parallel training nodes.
    
    Returns:
        Kedro Pipeline for training all three model variants
    """
    return pipeline(
        [
            # Base model training
            node(
                func=train_base_model,
                inputs=[
                    "train_dataset",
                    "val_dataset",
                    "scaler_y",
                    "params:base_model"
                ],
                outputs=["base_model", "base_model_metrics"],
                name="train_base_model_node",
                tags=["training", "base_model"]
            ),
            
            # Enriched model training
            node(
                func=train_enriched_model,
                inputs=[
                    "enriched_train_dataset",
                    "enriched_val_dataset",
                    "enriched_scaler_y",
                    "params:enriched_model"
                ],
                outputs=["enriched_model", "enriched_model_metrics"],
                name="train_enriched_model_node",
                tags=["training", "enriched_model"]
            ),
            
            # Kalman model training
            node(
                func=train_kalman_model,
                inputs=[
                    "kalman_train_dataset",
                    "kalman_val_dataset",
                    "kalman_scaler_y",
                    "params:kalman_model"
                ],
                outputs=["kalman_model", "kalman_model_metrics"],
                name="train_kalman_model_node",
                tags=["training", "kalman_model"]
            ),
        ]
    )
