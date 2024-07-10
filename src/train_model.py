""" #! ADD MODULE DOC STRING
"""

# Script to train machine learning model.

import yaml

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split

from clean_data import load_data
from ml.data import process_data
from ml.model import save_model, train_model

with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)


data = load_data(config["CLEANED_DATA_PATH"])

# Optional enhancement,
# use K-fold cross validation instead of a train-test split.
train, _ = train_test_split(data, test_size=0.20, random_state=42)


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=config["CAT_FEATURES"],
    label=config["LABEL"],
    training=True,
)

# # Train and save a model.
model = train_model(X_train, y_train)

save_model(model, file_name=config["MODEL_PATH"])
save_model(encoder, file_name=config["OHE_PATH"])
save_model(lb, file_name=config["LB_PATH"])
