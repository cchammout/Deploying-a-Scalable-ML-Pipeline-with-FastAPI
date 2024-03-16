from ml.model import train_model, inference
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
