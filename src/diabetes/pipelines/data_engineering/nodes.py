"""
Data Engineering Pipeline Nodes - Diabetes Prediction
======================================================
Pure Python functions: no I/O, no hardcoded paths.
All data is passed in and returned — Kedro handles I/O via the catalog.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def replace_zeros_with_nan(df: pd.DataFrame, zero_as_null_columns: list) -> pd.DataFrame:
    """
    Replace zeros with NaN in columns where 0 is biologically impossible.
    Glucose, BloodPressure, SkinThickness, Insulin and BMI cannot be 0.
    """
    df = df.copy()
    for col in zero_as_null_columns:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            logger.info("Column '%s': replacing %d zeros with NaN", col, zero_count)
            df[col] = df[col].replace(0, np.nan)
    return df


def impute_missing_values(
    df: pd.DataFrame,
    zero_as_null_columns: list,
    knn_imputer_neighbors: int,
) -> pd.DataFrame:
    """
    Impute missing values using KNNImputer.
    RobustScaler is applied before KNN and inverse-transformed after,
    exactly as done in the notebook.
    """
    df = df.copy()
    cols_to_impute = [c for c in zero_as_null_columns if c in df.columns]

    rs = RobustScaler()
    dff = pd.DataFrame(
        rs.fit_transform(df[cols_to_impute]),
        columns=cols_to_impute,
        index=df.index,
    )
    dff = pd.DataFrame(
        KNNImputer(n_neighbors=knn_imputer_neighbors).fit_transform(dff),
        columns=cols_to_impute,
        index=df.index,
    )
    dff = pd.DataFrame(
        rs.inverse_transform(dff),
        columns=cols_to_impute,
        index=df.index,
    )
    df[cols_to_impute] = dff
    logger.info("KNN imputation complete. NaN remaining: %d", df.isnull().sum().sum())
    return df


def cap_outliers(
    df: pd.DataFrame,
    zero_as_null_columns: list,
    outlier_q1: float,
    outlier_q3: float,
) -> pd.DataFrame:
    """
    Cap outliers using IQR-based thresholds (q1=0.05, q3=0.95).
    Values outside the limits are clipped, not removed.
    """
    df = df.copy()
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c.upper() != "OUTCOME"
    ]
    for col in numeric_cols:
        q1_val = df[col].quantile(outlier_q1)
        q3_val = df[col].quantile(outlier_q3)
        iqr = q3_val - q1_val
        low_limit = q1_val - 1.5 * iqr
        up_limit = q3_val + 1.5 * iqr
        before = ((df[col] < low_limit) | (df[col] > up_limit)).sum()
        df[col] = df[col].clip(lower=low_limit, upper=up_limit)
        if before > 0:
            logger.info("Column '%s': capped %d outliers", col, before)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering — exactly as done in the notebook.
    Creates: NEW_AGE_CAT, NEW_BMI, NEW_GLUCOSE, NEW_AGE_BMI_NOM,
             NEW_AGE_GLUCOSE_NOM, NEW_INSULIN_SCORE,
             NEW_GLUCOSE_INSULIN, NEW_GLUCOSE_PREGNANCIES.
    All column names are uppercased at the end.
    """
    df = df.copy()

    df["NEW_AGE_CAT"] = df["Age"].apply(lambda x: 0 if x < 50 else 1)

    df["NEW_BMI"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=["Underweight", "Healthy", "Overweight", "Obese"],
    )

    df["NEW_GLUCOSE"] = pd.cut(
        df["Glucose"],
        bins=[0, 70, 99, 125, np.inf],
        labels=["Low", "Normal", "Prediabetes", "Diabetes"],
    )

    df["NEW_AGE_BMI_NOM"] = "obesemature"
    df.loc[(df["BMI"] < 18.5) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "underweightmature"
    df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
    df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "healthymature"
    df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
    df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "overweightmature"
    df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
    df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

    df["NEW_AGE_GLUCOSE_NOM"] = "highmature"
    df.loc[(df["Glucose"] < 70) & (df["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
    df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
    df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
    df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
    df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
    df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
    df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

    df["NEW_INSULIN_SCORE"] = df["Insulin"].apply(
        lambda x: "Normal" if 16 <= x <= 166 else "Abnormal"
    )

    df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]
    df["NEW_GLUCOSE_PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

    df.columns = [col.upper() for col in df.columns]

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical/object columns except OUTCOME.
    Boolean columns are converted to int.
    """
    df = df.copy()
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c != "OUTCOME"
    ]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        logger.info("One-hot encoded: %s", cat_cols)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def align_columns(df: pd.DataFrame, reference_columns: list) -> pd.DataFrame:
    """
    Align a DataFrame's columns to match a reference list.

    This is CRITICAL for the inference pipeline: get_dummies on the
    inference dataset may produce fewer one-hot columns than training
    (if some categories don't appear in the smaller inference set).
    Missing columns are added with value 0; extra columns are dropped.
    """
    df = df.copy()
    missing = [c for c in reference_columns if c not in df.columns]
    if missing:
        logger.info("Adding %d missing columns (filled with 0): %s", len(missing), missing)
    df = df.reindex(columns=reference_columns, fill_value=0)
    return df


def fit_scaler(
    df: pd.DataFrame,
    columns_to_scale: list,
    target_column: str,
) -> RobustScaler:
    """
    Fit a RobustScaler on training data only — no data leakage.
    Returns the fitted scaler saved as a catalog artifact.
    """
    cols = [c for c in columns_to_scale if c in df.columns and c != target_column]
    scaler = RobustScaler()
    scaler.fit(df[cols])
    logger.info("Scaler fitted on %d columns", len(cols))
    return scaler


def transform_scaler(
    df: pd.DataFrame,
    scaler: RobustScaler,
    columns_to_scale: list,
    target_column: str,
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler. Never re-fits — avoids data leakage.
    """
    df = df.copy()
    cols = [c for c in columns_to_scale if c in df.columns and c != target_column]
    df[cols] = scaler.transform(df[cols])
    logger.info("Scaler applied to %d columns", len(cols))
    return df


def get_feature_columns(master_table: pd.DataFrame, target_column: str) -> list:
    """
    Extract the list of feature column names from the master table.
    Saved to catalog so the inference pipeline can use it to align columns.
    """
    target_col_upper = target_column.upper()
    cols = [c for c in master_table.columns if c != target_col_upper]
    logger.info("Feature columns saved: %d columns", len(cols))
    return cols
