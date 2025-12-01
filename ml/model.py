import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # TODO: Train a suitable model. RandomForestClassifier is a good starting point.
    # You might want to consider hyperparameter tuning for better performance.
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : trained machine learning model
        Trained machine learning model.
    X : pd.DataFrame
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # TODO: Implement the inference function.
    preds = model.predict(X)
    return preds


def compute_model_metrics(y, preds):
    """
    Computes the performance metrics when the model is trained and
    the predictions are made.

    Inputs
    ------
    y : np.array
        True labels.
    preds : np.array
        Predictions from the model.
    Returns
    -------
    precision, recall, fbeta
        Metrics.
    """
    fbeta = fbeta_score(y, preds, beta=0.5, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def performance_on_categorical_slice(
    model, data, categorical_feature, label_column, encoder, lb
):
    """
    Computes the performance metrics on a slice of the data, based on a categorical feature.

    Inputs
    ------
    model : trained machine learning model
        Trained machine learning model.
    data : pd.DataFrame
        Data used for evaluation.
    categorical_feature : str
        The name of the categorical feature to slice on.
    label_column : str
        The name of the label column.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder for categorical features.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer for the label column.

    Returns
    -------
    slice_metrics : dict
        A dictionary containing metrics for each slice of the categorical feature.
        Example: {"workclass_Private": {"precision": 0.8, "recall": 0.7, "fbeta": 0.75}, ...}
    """
    slice_metrics = {}
    unique_values = data[categorical_feature].unique()

    for value in unique_values:
        # Create a slice of the data for the current categorical value
        # Ensure we are working with a fresh copy to avoid SettingWithCopyWarning
        raw_sliced_data = data[data[categorical_feature] == value].copy()

        if raw_sliced_data.empty:
            continue

        # Get all categorical features from the original `data` for consistent processing.
        # Assuming we can derive them from the dataframe logic in a robust implementation.
        all_categorical_features = data.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        if label_column in all_categorical_features:
            all_categorical_features.remove(label_column)

        X_slice_processed, y_slice_processed, _, _ = process_data(
            raw_sliced_data,
            categorical_features=all_categorical_features,
            label=label_column,
            training=False,
            encoder=encoder,
            lb=lb
        )

        preds = inference(model, X_slice_processed)
        precision, recall, fbeta = compute_model_metrics(y_slice_processed, preds)

        slice_metrics[f"{categorical_feature}_{value}"] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta
        }

    return slice_metrics


def save_model(model, path):
    """
    Saves a trained model or any categorical encoders to a specified path.

    Inputs
    ------
    model : object
        The model or encoder object to save.
    path : str
        The file path to save the object to.
    Returns
    -------
    None
    """
    # TODO: Implement the save_model function.
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Loads a model or any categorical encoders from a specified path.

    Inputs
    ------
    path : str
        The file path from which to load the object.
    Returns
    -------
    object
        The loaded model or encoder object.
    """
    # TODO: Implement the load_model function.
    # Handle the case where path might be None (common during initial dev)
    if path is None:
        return None
    with open(path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object
