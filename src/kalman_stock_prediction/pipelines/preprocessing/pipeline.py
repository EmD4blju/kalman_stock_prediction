from kedro.pipeline import Pipeline, node
from .nodes import apply_kalman_filter, enrich_dataset, reformat_periodic_to_supervised_data, scale_dataframe, split_dataframe


def create_preprocessing_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            #~ Regular dataset path
            node(
                func=reformat_periodic_to_supervised_data,
                inputs=["raw_dataset", "params:target_column", "params:timesteps"],
                outputs="supervised_dataset",
                name="reformat_periodic_to_supervised_data_node",
            ),
            node(
                func=scale_dataframe,
                inputs=["supervised_dataset", "params:scaling_method", "params:target_column"],
                outputs=["scaled_dataset", "scaler_X", "scaler_y"],
                name="scale_dataframe_node",
            ),
            node(
                func=split_dataframe,
                inputs=["scaled_dataset", "params:val_size", "params:test_size"],
                outputs=["train_dataset", "val_dataset", "test_dataset"],
                name="split_dataframe_node",
            ),
            #~ Enriched dataset path
            node(
                func=enrich_dataset,
                inputs=["raw_dataset", "params:target_column", "params:rsi_window", "params:bb_window", "params:bb_window_dev"],
                outputs="raw_enriched_dataset",
                name="enrich_dataset_node",
            ),
            node(
                func=reformat_periodic_to_supervised_data,
                inputs=["raw_enriched_dataset", "params:target_column", "params:timesteps"],
                outputs="enriched_supervised_dataset",
                name="reformat_periodic_to_supervised_data_enriched_node",
            ),
            node(
                func=scale_dataframe,
                inputs=["enriched_supervised_dataset", "params:scaling_method", "params:target_column"],
                outputs=["enriched_scaled_dataset", "enriched_scaler_X", "enriched_scaler_y"],
                name="scale_dataframe_enriched_node",
            ),
            node(
                func=split_dataframe,
                inputs=["enriched_scaled_dataset", "params:val_size", "params:test_size"],
                outputs=["enriched_train_dataset", "enriched_val_dataset", "enriched_test_dataset"],
                name="split_dataframe_enriched_node",
            ),
            #~ Kalman filtered dataset path
            node(
                func=apply_kalman_filter,
                inputs=["raw_dataset", "params:target_column", "params:kalman_F", "params:kalman_H", "params:kalman_P", "params:kalman_R", "params:kalman_Q"],
                outputs="kalman_dataset",
                name="apply_kalman_filter_node",
            ),
            node(
                func=reformat_periodic_to_supervised_data,
                inputs=["kalman_dataset", "params:target_column", "params:timesteps"],
                outputs="kalman_supervised_dataset",
                name="reformat_periodic_to_supervised_data_kalman_node",
            ),
            node(
                func=scale_dataframe,
                inputs=["kalman_supervised_dataset", "params:scaling_method", "params:target_column"],
                outputs=["kalman_scaled_dataset", "kalman_scaler_X", "kalman_scaler_y"],
                name="scale_dataframe_kalman_node",
            ),
            node(
                func=split_dataframe,
                inputs=["kalman_scaled_dataset", "params:val_size", "params:test_size"],
                outputs=["kalman_train_dataset", "kalman_val_dataset", "kalman_test_dataset"],
                name="split_dataframe_kalman_node",
            ),
        ]
    )
