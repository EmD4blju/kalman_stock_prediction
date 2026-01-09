"""Project pipelines."""


from kedro.pipeline import Pipeline
from kalman_stock_prediction.pipelines.preprocessing.pipeline import create_preprocessing_pipeline
from kalman_stock_prediction.pipelines.modelling.pipeline import create_modelling_pipeline
from kalman_stock_prediction.pipelines.evaluation.pipeline import create_evaluation_pipeline
from kalman_stock_prediction.pipelines.tuning.pipeline import create_tuning_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    preprocessing_pipeline = create_preprocessing_pipeline()
    modelling_pipeline = create_modelling_pipeline()
    evaluation_pipeline = create_evaluation_pipeline()
    tuning_pipeline = create_tuning_pipeline()
    
    return {
        "preprocessing": preprocessing_pipeline,
        "modelling": modelling_pipeline,
        "evaluation": evaluation_pipeline,
        "tuning": tuning_pipeline,
        "__default__": preprocessing_pipeline + modelling_pipeline + evaluation_pipeline,
    }
    
