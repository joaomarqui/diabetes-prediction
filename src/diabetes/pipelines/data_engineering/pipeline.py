"""
Data Engineering Pipeline - Diabetes Prediction
================================================
DAG:
  raw_data
    -> replace_zeros_with_nan
    -> impute_missing_values
    -> cap_outliers
    -> create_features
    -> encode_categorical_features
    -> fit_scaler
    -> transform_scaler  -> master_table
    -> get_feature_columns -> feature_columns  (saved for inference alignment)
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    cap_outliers,
    create_features,
    encode_categorical_features,
    fit_scaler,
    get_feature_columns,
    impute_missing_values,
    replace_zeros_with_nan,
    transform_scaler,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=replace_zeros_with_nan,
                inputs=["raw_diabetes_modelling", "params:zero_as_null_columns"],
                outputs="diabetes_zeros_replaced",
                name="replace_zeros_with_nan_node",
            ),
            node(
                func=impute_missing_values,
                inputs=[
                    "diabetes_zeros_replaced",
                    "params:zero_as_null_columns",
                    "params:knn_imputer_neighbors",
                ],
                outputs="diabetes_imputed",
                name="impute_missing_values_node",
            ),
            node(
                func=cap_outliers,
                inputs=[
                    "diabetes_imputed",
                    "params:zero_as_null_columns",
                    "params:outlier_q1",
                    "params:outlier_q3",
                ],
                outputs="cleaned_diabetes_data",
                name="cap_outliers_node",
            ),
            node(
                func=create_features,
                inputs="cleaned_diabetes_data",
                outputs="diabetes_with_features",
                name="create_features_node",
            ),
            node(
                func=encode_categorical_features,
                inputs="diabetes_with_features",
                outputs="diabetes_encoded",
                name="encode_categorical_features_node",
            ),
            node(
                func=fit_scaler,
                inputs=[
                    "diabetes_encoded",
                    "params:columns_to_scale",
                    "params:target_column",
                ],
                outputs="production_scaler",
                name="fit_scaler_node",
            ),
            node(
                func=transform_scaler,
                inputs=[
                    "diabetes_encoded",
                    "production_scaler",
                    "params:columns_to_scale",
                    "params:target_column",
                ],
                outputs="master_table",
                name="transform_scaler_node",
            ),
            node(
                func=get_feature_columns,
                inputs=["master_table", "params:target_column"],
                outputs="feature_columns",
                name="get_feature_columns_node",
            ),
        ]
    )
