from ml.data import process_data
from ml.model import load_model
from src.train_model import cat_features

encoder = load_model("./model/onehotencoder.pkl")


# def test_process_data(df):
#     X_test, y_test, _, _ = process_data(
#         df,
#         categorical_features=cat_features,
#         training=False,
#         encoder=encoder,
#     )
