"""
Inference Pipeline Nodes - Diabetes Prediction
===============================================
Reuses data engineering nodes — no fitting, only transform.
Two new functions: prepare_inference_data and predict.

Key fix: align_columns ensures the inference DataFrame has
exactly the same columns as the training data, in the same order.
This is necessary because get_dummies on a smaller inference dataset
may not produce all the one-hot columns that exist in training.
"""

import logging

import pandas as pd

from diabetes.pipelines.data_engineering.nodes import (
    align_columns,
    cap_outliers,
    create_features,
    encode_categorical_features,
    impute_missing_values,
    replace_zeros_with_nan,
    transform_scaler,
)

logger = logging.getLogger(__name__)


def prepare_inference_data(
    raw_inference: pd.DataFrame,
    zero_as_null_columns: list,
    knn_imputer_neighbors: int,
    outlier_q1: float,
    outlier_q3: float,
    scaler,
    columns_to_scale: list,
    target_column: str,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Apply full data engineering transformation to inference data.
    Reuses all data engineering node functions — no code duplication.
    No fitting: only transforms with production artifacts.

    feature_columns: the exact column list from training, used to
    align the inference DataFrame after one-hot encoding.
    """
    df = replace_zeros_with_nan(raw_inference, zero_as_null_columns)
    df = impute_missing_values(df, zero_as_null_columns, knn_imputer_neighbors)
    df = cap_outliers(df, zero_as_null_columns, outlier_q1, outlier_q3)
    df = create_features(df)
    df = encode_categorical_features(df)

    # Align columns to match training — fills missing dummies with 0
    df = align_columns(df, feature_columns + [target_column.upper()])

    df = transform_scaler(df, scaler, columns_to_scale, target_column)

    logger.info("Inference data prepared. Shape: %s", df.shape)
    return df


def predict(model, inference_data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Run predictions on the prepared inference data.
    Returns DataFrame with predictions and probability scores.
    """
    target_col_upper = target_column.upper()
    X = inference_data.drop(columns=[target_col_upper], errors="ignore")

    predictions = model.predict(X)
    probabilities = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else predictions
    )

    result = pd.DataFrame(
        {
            "prediction": predictions,
            "probability_diabetes": probabilities,
        }
    )

    positive = int(predictions.sum())
    logger.info(
        "Inference complete: %d records, %d predicted positive (%.1f%%)",
        len(predictions),
        positive,
        100 * positive / len(predictions),
    )
    return result
