import json
import requests

# Send a GET request
url = "http://127.0.0.1:8000"
response_get = requests.get(url)

# Print the status code and welcome message for GET request
print("GET Request:")
print("Status Code:", response_get.status_code)
print("Welcome Message:", response_get.text)
print()

# Prepare data for POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request
response_post = requests.post(url + "/data/", json=data)

# Print the status code and result for POST request
print("POST Request:")
print("Status Code:", response_post.status_code)
print("Result:", response_post.json()["result"])
