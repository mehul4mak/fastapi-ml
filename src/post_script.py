"""
Client Script for Making Predictions

This script sends a sample input dictionary to a specified URL
for making predictions.It prints the response status code
and content.
"""

from typing import Any, Dict

import requests

# Define the sample input dictionary
sample_dict: Dict[str, Any] = {
    "workclass": "state_gov",
    "education": "bachelors",
    "marital_status": "never_married",
    "occupation": "adm_clerical",
    "relationship": "not_in_family",
    "race": "white",
    "sex": "male",
    "native_country": "united_states",
    "age": 39,
    "fnlwgt": 77516,
    "education_num": 13,
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
}

# Define the URL for the prediction API
url: str = "https://udacity-c3-project.herokuapp.com/predict"

# Send a POST request to the API with the sample input dictionary
post_response = requests.post(url, json=sample_dict)

# Print the response status code and content
print(post_response.status_code)
print(post_response.content)
