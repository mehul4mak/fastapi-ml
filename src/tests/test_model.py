import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)


def test_train_model(df, cat_features, label):

    train, _ = train_test_split(df, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    model = train_model(X_train, y_train)

    assert model is not None
    assert isinstance(model, BaseEstimator)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict_proba")


def test_inference(df, cat_features, label):

    train, _ = train_test_split(df, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    model = train_model(X_train, y_train)

    preds = inference(model, X_train)

    assert preds.shape[0] == X_train.shape[0]


def test_compute_model_metrics(df, cat_features, label):

    train, _ = train_test_split(df, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    model = train_model(X_train, y_train)

    preds = inference(model, X_train)

    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_compute_model_metrics_simulation():
    # Simulate data
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1]

    # Calculate metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Assert exact values (assuming known correct values)
    assert precision == 1
    assert recall == 0.6666666666666666
    assert fbeta == 0.8
