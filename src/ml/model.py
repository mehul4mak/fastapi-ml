"""Model Python File

This module provides utilities for training, evaluating,
saving, and loading machine learning models.
"""

import pickle
from typing import Callable, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

# pylint: disable=C0103


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
    """
    Trains a machine learning model and returns it.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series
        Labels.

    Returns
    -------
    BaseEstimator in this case RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(
    y: pd.Series, preds: pd.Series
) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Parameters
    ----------
    y : pd.Series
        Known labels, binarized.
    preds : pd.Series
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    """
    Run model inferences and return the predictions.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.
    X : pd.DataFrame
        Data used for prediction.

    Returns
    -------
    preds : pd.Series
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model: BaseEstimator, file_name: str = "model.pkl"):
    """
    Save sklearn model into pickle format.

    Parameters
    ----------
    model : BaseEstimator
        Sklearn Model
    file_name : str, optional
        File name with path to save the model in dir, by default "model.pkl"
    """
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name: str) -> BaseEstimator:
    """
    Load sklearn model from given file path/name.

    Parameters
    ----------
    file_name : str
        File path where model is stored

    Returns
    -------
    BaseEstimator
        Loaded machine learning model.
    """
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


ProcessDataCallable = Callable[
    [pd.DataFrame, list, str, bool, object, object],
    Tuple[pd.DataFrame, pd.Series, object, object],
]


def data_slice_based_model_metrics(
    data: pd.DataFrame,
    model: BaseEstimator,
    cat_features: list,
    label: str,
    encoder: object,
    lb: object,
    process_data: ProcessDataCallable,
):
    """
    Compute model metrics on data data_slices based on categorical features.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    model : BaseEstimator
        Trained machine learning model.
    cat_features : list
        List of categorical features.
    label : str
        Target label column name.
    encoder : object
        Encoder used for categorical features.
    lb : object
        Label binarizer.
    process_data : ProcessDataCallable
        Function to process the data.

    Returns
    -------
    None
    """
    data_slices = {}

    for col in cat_features:
        print(col, data[col].nunique())
        print(">" * 10)
        for i, data_slice in enumerate(data[col].unique()):
            print(i + 1, data_slice)

            subset_df = data[data[col] == data_slice].copy()

            X_test, y_test, encoder, lb = process_data(
                subset_df,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            y_preds = inference(model, X_test)

            data_slices[str(col) + "_" + str(data_slice)] = dict(
                zip(
                    ("precision", "recall", "fbeta"),
                    compute_model_metrics(y_test, y_preds),
                )
            )

        print("-" * 50)

    data_slice_df = pd.DataFrame(data_slices).T
    data_slice_df.index.name = "category"
    data_slice_df.to_csv("output.txt")
