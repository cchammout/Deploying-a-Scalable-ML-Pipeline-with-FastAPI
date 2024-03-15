import pytest
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Test if train_model returns a model object
def test_train_model_returns_model_object():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

# Test if inference returns numpy array of predictions
def test_inference_returns_numpy_array():
    model = RandomForestClassifier()
    X_train = np.array([[1, 2], [3, 4]])  # Sample training data
    y_train = np.array([0, 1])  # Sample labels
    model.fit(X_train, y_train)  # Train the model
    X_test = np.array([[1, 2]])  # Sample test data
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)

# Add a test to verify if the ML model uses the expected algorithm
def test_model_uses_expected_algorithm():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

# Add your own tests based on your requirements and functions

# Run pytest to execute the tests
# pytest will automatically discover and run all the functions that start with 'test_'
# Run pytest in your terminal from your project directory
# pytest


