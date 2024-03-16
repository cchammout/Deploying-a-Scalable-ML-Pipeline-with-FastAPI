import requests

# Send a GET request
url_get = "http://127.0.0.1:8000/"
response_get = requests.get(url_get)

# Print the status code and welcome message from the GET request
print("GET Request:")
print("Status Code:", response_get.status_code)
print("Welcome Message:", response_get.json()["message"])

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
url_post = "http://127.0.0.1:8000/data/"
response_post = requests.post(url_post, json=data)

# Print the status code and result from the POST request
print("\nPOST Request:")
print("Status Code:", response_post.status_code)
try:
    result = response_post.json().get("result", "No result found")
    print("Result:", result)
except ValueError:
    print("Error: Failed to decode JSON response")
    print("Response Content:", response_post.content.decode())
    