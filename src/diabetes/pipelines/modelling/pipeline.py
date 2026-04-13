"""
Modelling Pipeline - Diabetes Prediction
=========================================
DAG: master_table -> split -> train -> evaluate -> production_model
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "master_table",
                    "params:target_column",
                    "params:test_size",
                    "params:random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:model_class_path",
                    "params:model_init_args",
                ],
                outputs="production_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["production_model", "X_test", "y_test"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
