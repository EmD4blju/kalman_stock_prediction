"""Pipeline for model evaluation."""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_base_model,
    evaluate_enriched_model,
    evaluate_kalman_model_filtered,
    evaluate_kalman_model_original,
    plot_base_model_learning_curves,
    plot_enriched_model_learning_curves,
    plot_kalman_model_learning_curves
)


def create_evaluation_pipeline(**kwargs) -> Pipeline:
    """
    Create the evaluation pipeline with four evaluation nodes.
    Kalman model is evaluated on both filtered and original data.
    Each node outputs metrics and visualization plots.
    
    Returns:
        Kedro Pipeline for evaluating all model variants
    """
    return pipeline(
        [
            # Base model evaluation
            node(
                func=evaluate_base_model,
                inputs=[
                    "test_dataset",
                    "base_model",
                    "scaler_y",
                    "params:evaluation"
                ],
                outputs=[
                    "base_model_test_metrics",
                    "base_model_actual_vs_predicted_plot",
                    "base_model_error_distribution_plot"
                ],
                name="evaluate_base_model_node",
                tags=["evaluation", "base_model"]
            ),
            
            # Base model learning curves
            node(
                func=plot_base_model_learning_curves,
                inputs="base_model_metrics",
                outputs="base_model_learning_curves_plot",
                name="plot_base_model_learning_curves_node",
                tags=["visualization", "base_model", "learning_curves"]
            ),
            
            # Enriched model evaluation
            node(
                func=evaluate_enriched_model,
                inputs=[
                    "enriched_test_dataset",
                    "enriched_model",
                    "enriched_scaler_y",
                    "params:evaluation"
                ],
                outputs=[
                    "enriched_model_test_metrics",
                    "enriched_model_actual_vs_predicted_plot",
                    "enriched_model_error_distribution_plot"
                ],
                name="evaluate_enriched_model_node",
                tags=["evaluation", "enriched_model"]
            ),
            
            # Enriched model learning curves
            node(
                func=plot_enriched_model_learning_curves,
                inputs="enriched_model_metrics",
                outputs="enriched_model_learning_curves_plot",
                name="plot_enriched_model_learning_curves_node",
                tags=["visualization", "enriched_model", "learning_curves"]
            ),
            
            # Kalman model evaluation on filtered data
            node(
                func=evaluate_kalman_model_filtered,
                inputs=[
                    "kalman_test_dataset",
                    "kalman_model",
                    "kalman_scaler_y",
                    "params:evaluation"
                ],
                outputs=[
                    "kalman_model_filtered_test_metrics",
                    "kalman_model_filtered_actual_vs_predicted_plot",
                    "kalman_model_filtered_error_distribution_plot"
                ],
                name="evaluate_kalman_model_filtered_node",
                tags=["evaluation", "kalman_model", "filtered"]
            ),
            
            # Kalman model evaluation on original data
            node(
                func=evaluate_kalman_model_original,
                inputs=[
                    "test_dataset",
                    "kalman_model",
                    "scaler_y",
                    "params:evaluation"
                ],
                outputs=[
                    "kalman_model_original_test_metrics",
                    "kalman_model_original_actual_vs_predicted_plot",
                    "kalman_model_original_error_distribution_plot"
                ],
                name="evaluate_kalman_model_original_node",
                tags=["evaluation", "kalman_model", "original"]
            ),
            
            # Kalman model learning curves
            node(
                func=plot_kalman_model_learning_curves,
                inputs="kalman_model_metrics",
                outputs="kalman_model_learning_curves_plot",
                name="plot_kalman_model_learning_curves_node",
                tags=["visualization", "kalman_model", "learning_curves"]
            ),
        ]
    )
