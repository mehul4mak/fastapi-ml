""" #! ADD MODULE DOC STRING
"""

from sklearn.model_selection import train_test_split

from clean_data import load_data
from ml.data import process_data
from ml.model import compute_model_metrics, inference, load_model

data = load_data("./data/cleaned_census.csv")

# Optional enhancement,
# use K-fold cross validation instead of a train-test split.
_, test = train_test_split(data, test_size=0.01, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

model = load_model(file_name="./model/model.pkl")
encoder = load_model(file_name="./model/onehotencoder.pkl")
lb = load_model(file_name="./model/labelbinarizer.pkl")

X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

preds = inference(model, X_test)
print(y_test, preds)


print(compute_model_metrics(y_test, preds))
