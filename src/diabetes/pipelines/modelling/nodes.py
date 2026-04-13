"""
Modelling Pipeline Nodes - Diabetes Prediction
===============================================
Pure Python functions: train, evaluate, and optimize the model.
Model class is loaded dynamically from parameters — swap via YAML.
"""

import importlib
import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _load_class(class_path: str):
    """Dynamically import a class from its dotted path string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def split_data(
    master_table: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
):
    """
    Split master_table into train and test sets.
    y_train and y_test are DataFrames (not Series) so Kedro
    can persist them correctly via the catalog.
    """
    target_col_upper = target_column.upper()
    X = master_table.drop(columns=[target_col_upper])
    y = master_table[[target_col_upper]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Data split: train=%d, test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_class_path: str, model_init_args: dict):
    """
    Train the classifier dynamically loaded from parameters.yml.
    y_train is squeezed from DataFrame to Series for sklearn.
    """
    cls = _load_class(model_class_path)
    model = cls(**model_init_args)
    model.fit(X_train, y_train.squeeze())
    logger.info("Model trained: %s", model_class_path)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model on the test set.
    Returns accuracy, recall, precision, f1, auc.
    """
    y_true = y_test.squeeze()
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "auc": round(roc_auc_score(y_true, y_proba), 4),
    }

    logger.info("Model evaluation: %s", metrics)
    return metrics
