from kedro.pipeline import Pipeline, node
from .nodes import apply_kalman_filter, enrich_dataset, reformat_to_supervised, reformat_enriched_to_supervised, fit_scalers, apply_scalers, split_dataframe


def create_preprocessing_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # ==================================================================
            # ~ Regular dataset path
            # ==================================================================
            node(
                func=reformat_to_supervised,
                inputs=["raw_dataset", "params:target_column", "params:timesteps"],
                outputs="supervised_dataset",
                name="reformat_periodic_to_supervised_data_node",
            ),
            node(
                func=split_dataframe,
                inputs=["supervised_dataset", "params:val_size", "params:test_size"],
                outputs=["raw_train_dataset", "raw_val_dataset", "raw_test_dataset"],
                name="split_dataframe_node",
            ),
            node(
                func=fit_scalers,
                inputs=["raw_train_dataset", "params:scaling_method", "params:target_column"],
                outputs=["scaler_X", "scaler_y"],
                name="fit_scalers_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_train_dataset", "scaler_X", "scaler_y", "params:target_column"],
                outputs="train_dataset",
                name="apply_scalers_to_train_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_val_dataset", "scaler_X", "scaler_y", "params:target_column"],
                outputs="val_dataset",
                name="apply_scalers_to_val_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_test_dataset", "scaler_X", "scaler_y", "params:target_column"],
                outputs="test_dataset",
                name="apply_scalers_to_test_node",
            ),
            # ==================================================================
            # ~ Enriched dataset path
            # ==================================================================
            node(
                func=enrich_dataset,
                inputs=["raw_dataset", "params:target_column", "params:rsi_window", "params:bb_window", "params:bb_window_dev"],
                outputs="raw_enriched_dataset",
                name="enrich_dataset_node",
            ),
            node(
                func=reformat_enriched_to_supervised,
                inputs=["raw_enriched_dataset", "params:target_column", "params:timesteps"],
                outputs="enriched_supervised_dataset",
                name="reformat_periodic_to_supervised_data_enriched_node",
            ),
            node(
                func=split_dataframe,
                inputs=["enriched_supervised_dataset", "params:val_size", "params:test_size"],
                outputs=["raw_enriched_train_dataset", "raw_enriched_val_dataset", "raw_enriched_test_dataset"],
                name="split_dataframe_enriched_node",
            ),
            node(
                func=fit_scalers,
                inputs=["raw_enriched_train_dataset", "params:scaling_method", "params:target_column"],
                outputs=["enriched_scaler_X", "enriched_scaler_y"],
                name="fit_scalers_enriched_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_enriched_train_dataset", "enriched_scaler_X", "enriched_scaler_y", "params:target_column"],
                outputs="enriched_train_dataset",
                name="apply_scalers_to_enriched_train_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_enriched_val_dataset", "enriched_scaler_X", "enriched_scaler_y", "params:target_column"],
                outputs="enriched_val_dataset",
                name="apply_scalers_to_enriched_val_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_enriched_test_dataset", "enriched_scaler_X", "enriched_scaler_y", "params:target_column"],
                outputs="enriched_test_dataset",
                name="apply_scalers_to_enriched_test_node",
            ),
            # ==================================================================
            # ~ Kalman filtered dataset path
            # ==================================================================
            node(
                func=apply_kalman_filter,
                inputs=["raw_dataset", "params:target_column", "params:kalman_F", "params:kalman_H", "params:kalman_P", "params:kalman_R", "params:kalman_Q"],
                outputs="kalman_dataset",
                name="apply_kalman_filter_node",
            ),
            node(
                func=reformat_to_supervised,
                inputs=["kalman_dataset", "params:target_column", "params:timesteps"],
                outputs="kalman_supervised_dataset",
                name="reformat_periodic_to_supervised_data_kalman_node",
            ),
            node(
                func=split_dataframe,
                inputs=["kalman_supervised_dataset", "params:val_size", "params:test_size"],
                outputs=["raw_kalman_train_dataset", "raw_kalman_val_dataset", "raw_kalman_test_dataset"],
                name="split_dataframe_kalman_node",
            ),
            node(
                func=fit_scalers,
                inputs=["raw_kalman_train_dataset", "params:scaling_method", "params:target_column"],
                outputs=["kalman_scaler_X", "kalman_scaler_y"],
                name="fit_scalers_kalman_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_kalman_train_dataset", "kalman_scaler_X", "kalman_scaler_y", "params:target_column"],
                outputs="kalman_train_dataset",
                name="apply_scalers_to_kalman_train_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_kalman_val_dataset", "kalman_scaler_X", "kalman_scaler_y", "params:target_column"],
                outputs="kalman_val_dataset",
                name="apply_scalers_to_kalman_val_node",
            ),
            node(
                func=apply_scalers,
                inputs=["raw_kalman_test_dataset", "kalman_scaler_X", "kalman_scaler_y", "params:target_column"],
                outputs="kalman_test_dataset",
                name="apply_scalers_to_kalman_test_node",
            ),
        ]
    )
