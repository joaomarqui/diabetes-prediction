"""
Inference Pipeline - Diabetes Prediction
=========================================
DAG:
  raw_inference_data + production_scaler + production_model + feature_columns
    -> prepare_inference_data
    -> predict
    -> inference_predictions
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict, prepare_inference_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_inference_data,
                inputs=[
                    "raw_inference_data",
                    "params:zero_as_null_columns",
                    "params:knn_imputer_neighbors",
                    "params:outlier_q1",
                    "params:outlier_q3",
                    "production_scaler",
                    "params:columns_to_scale",
                    "params:target_column",
                    "feature_columns",
                ],
                outputs="prepared_inference_data",
                name="prepare_inference_data_node",
            ),
            node(
                func=predict,
                inputs=[
                    "production_model",
                    "prepared_inference_data",
                    "params:target_column",
                ],
                outputs="inference_predictions",
                name="predict_node",
            ),
        ]
    )
