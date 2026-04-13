"""
Pipeline Registry - Diabetes Prediction
========================================
Registers all pipelines so Kedro can discover them.
Run individual pipelines:
    kedro run --pipeline data_engineering
    kedro run --pipeline modelling
    kedro run --pipeline inference
Run all:
    kedro run
"""

from kedro.pipeline import Pipeline

from diabetes.pipelines.data_engineering import create_pipeline as de_pipeline
from diabetes.pipelines.inference import create_pipeline as inf_pipeline
from diabetes.pipelines.modelling import create_pipeline as mod_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_engineering = de_pipeline()
    modelling = mod_pipeline()
    inference = inf_pipeline()

    return {
        "data_engineering": data_engineering,
        "modelling": modelling,
        "inference": inference,
        # Default: run data_engineering + modelling end-to-end
        "__default__": data_engineering + modelling,
    }
