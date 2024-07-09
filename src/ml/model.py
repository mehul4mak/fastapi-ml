"""Model Python Fil"""

import pickle
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


class MLModel(ABC):
    """ML Model basic representation"""

    @abstractmethod
    def fit(self):
        """Model Training or Fitting function"""
        pass

    @abstractmethod
    def predict(self):
        """Model Prediction Function"""
        pass


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
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


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model: MLModel, file_name: str = "model.pkl"):
    """Save sklearn model into pickle format.

    Parameters
    ----------
    model : BaseEstimator
        Sklearn Model
    file_name : str, optional
        file_name with path to save the model in dir, by default "model.pkl"
    """
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name: str) -> MLModel:
    """Load sklearn model from given file path/name.

    Parameters
    ----------
    file_name : str
        File path where model is stored

    Returns
    -------
    _type_
        _description_
    """
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model
