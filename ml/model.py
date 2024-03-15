import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

# Train a machine learning model and return it
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Example: Using Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Run model inferences and return the predictions
def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

# Serialize model or any categorical encoders to a file
def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

# Load model or any categorical encoders from a file
def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    with open(path, 'rb') as f:
        return pickle.load(f)

# Compute model metrics
def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

# Compute model metrics on a slice of the data
def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """ Computes the model metrics on a slice of the data specified by a column name and value.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features.
    label : str
        Name of the label column in `data`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model : object
        Model used for the task.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # Filter data based on slice value
    sliced_data = data[data[column_name] == slice_value].copy()
    
    # Process sliced data
    X_slice, y_slice, _, _ = process_data(sliced_data, categorical_features, label, training=False, encoder=encoder, lb=lb)
    
    # Get predictions
    preds = inference(model, X_slice)
    
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    
    return precision, recall, fbeta
