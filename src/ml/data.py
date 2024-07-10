"""
Data Processing Module

This module provides a function to process data for machine learning pipelines.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# pylint: disable=C0103


def process_data(
    X: pd.DataFrame,
    categorical_features: List[str],
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]:
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and
    a label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add
    in functionality that scales the continuous data.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features and label.
        Columns in `categorical_features`
    categorical_features : List[str]
        List containing the names of the categorical features (default=[])
    label : Optional[str]
        Name of the label column in `X`. If None, then an empty array will
        be returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : Optional[OneHotEncoder]
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : Optional[LabelBinarizer]
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.ndarray
        Processed data.
    y : np.ndarray
        Processed labels if labeled=True, otherwise empty np.ndarray.
    encoder : OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features, axis=1)

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
