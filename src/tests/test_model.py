"""Tests
"""

from sklearn.base import BaseEstimator, TransformerMixin

from ml.model import compute_model_metrics, inference, train_model

# def test_train_model(X, y):
#     model = train_model(X, y)
#     assert isinstance(model, BaseEstimator)
#     assert isinstance(model, TransformerMixin)


# def test_inference(model, X):
#     y_pred = inference(model, X)
#     assert y_pred.shape[0] == X.shape[0]
#     assert y.size() >= 0


# def test_compute_model_metrics(y, y_pred):
#     precision, recall, fbeta = compute_model_metrics(y, y_pred)
#     assert 0 <= precision <= 1
#     assert 0 <= recall <= 1
#     assert 0 <= fbeta <= 1

#  df = pd.DataFrame(
#         [
#             [
#                 64,
#                 "Private",
#                 21174,
#                 "HS-grad",
#                 9,
#                 "Married-civ-spouse",
#                 "Exec-managerial",
#                 "Husband",
#                 "White",
#                 "Male",
#                 0,
#                 0,
#                 40,
#                 "United-States",
#             ]
#         ],
#         columns=[
#             "age",
#             "workclass",
#             "fnlgt",
#             "education",
#             "education_num",
#             "marital_status",
#             "occupation",
#             "relationship",
#             "race",
#             "sex",
#             "capital_gain",
#             "capital_loss",
#             "hours_per_week",
#             "native_country",
#         ],
#     )
