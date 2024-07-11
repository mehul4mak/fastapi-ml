"""
Train Machine Learning Model

This script trains a machine learning model using the specified configuration.
It processes the data, trains the model, and saves the trained model
 and encoders.
"""

import yaml
from sklearn.model_selection import train_test_split

from clean_data import load_data
from ml.data import process_data
from ml.model import data_slice_based_model_metrics, save_model, train_model

# Load configuration from yaml file
with open("config.yaml", "rb") as f:
    config = yaml.safe_load(f)

# Load and process data
data = load_data(config["CLEANED_DATA_PATH"])

# Optional enhancement: use K-fold cross validation
# instead of a train-test split.
train, _ = train_test_split(data, test_size=0.20, random_state=42)

# Process the training data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=config["CAT_FEATURES"],
    label=config["LABEL"],
    training=True,
)

# Train and save the model.
model = train_model(X_train, y_train)
save_model(model, file_name=config["MODEL_PATH"])
save_model(encoder, file_name=config["OHE_PATH"])
save_model(lb, file_name=config["LB_PATH"])


# Compute metrics on each slices
data_slice_based_model_metrics(
    data,
    model,
    config["CAT_FEATURES"],
    label=config["LABEL"],
    encoder=encoder,
    lb=lb,
    process_data=process_data,
)
