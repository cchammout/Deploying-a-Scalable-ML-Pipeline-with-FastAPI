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
    X_test = np.array([[1, 2]])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)

# Test if compute_model_metrics returns expected precision, recall, and fbeta values
def test_compute_model_metrics_returns_expected_values():
    y_true = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    p, r, fb = compute_model_metrics(y_true, preds)
    assert round(p, 4) == 0.6667
    assert round(r, 4) == 0.6667
    assert round(fb, 4) == 0.6667

# Run pytest to execute the tests
# pytest will automatically discover and run all the functions that start with 'test_'
# Run pytest in your terminal from your project directory
# pytest

