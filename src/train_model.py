""" #! ADD MODULE DOC STRING
"""

# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split

from clean_data import load_data
from ml.data import process_data
from ml.model import save_model, train_model

data = load_data("data/cleaned_census.csv")

# Optional enhancement,
# use K-fold cross validation instead of a train-test split.
train, _ = train_test_split(data, test_size=0.20, random_state=42)

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


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# # Train and save a model.
model = train_model(X_train, y_train)

save_model(model, file_name="./model/model.pkl")
save_model(encoder, file_name="./model/onehotencoder.pkl")
save_model(lb, file_name="./model/labelbinarizer.pkl")
