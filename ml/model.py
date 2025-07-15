import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier # A good general-purpose classifier
import pandas as pd


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


def performance_on_categorical_slice(model, data, categorical_feature, label_column, encoder, lb):
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
        sliced_data = data[data[categorical_feature] == value]

        if sliced_data.empty:
            continue

        # Separate features and labels for the slice
        X_slice = sliced_data.drop(columns=[label_column])
        y_slice = sliced_data[label_column]

        # Process the slice data using the same encoder and lb used for training
        # This requires `process_data` or similar logic, but for simplicity here
        # we'll assume X_slice is already preprocessed or can be processed on the fly
        # if `process_data` handles it without needing X_train/y_train.
        # For this function, we'll assume X_slice needs to be encoded.
        # The `process_data` function typically handles the encoding.
        # You would call `process_data` with `training=False` and pass the `encoder` and `lb`.

        # For this exercise, let's assume `data` passed in is the preprocessed data (X and y)
        # and that the categorical features are still distinguishable or we re-encode just the features.
        # However, typically you'd re-run `process_data` on `sliced_data`
        # with the existing `encoder` and `lb`.
        
        # A more robust implementation would involve re-processing the raw sliced_data
        # with the original encoder and label binarizer.
        # For now, let's simplify and assume the `data` passed here is the original raw data
        # and we need to re-encode it for inference, using the provided encoder and lb.
        
        # This part assumes `data` is the original, unprocessed DataFrame.
        # You will need to import `process_data` from `ml.data` if you use it.
        # from ml.data import process_data

        # Example if you need to process the slice:
        # X_slice_processed, _, _, _ = process_data(
        #     sliced_data,
        #     categorical_features=[col for col in data.columns if data[col].dtype == 'object' and col != label_column],
        #     label=label_column,
        #     training=False,
        #     encoder=encoder,
        #     lb=lb
        # )
        # y_slice_processed = lb.transform(y_slice.values).ravel()

        # For the purpose of this function, assuming `data` in the input
        # is already processed (features are numerical, labels are numerical).
        # If not, you need to call `process_data` on `sliced_data`
        # to get `X_slice_processed` and `y_slice_processed`.

        # As a hint, the prompt suggests this function takes `data` as input,
        # which likely implies it's the full processed dataframe.
        # Let's assume `data` is already processed (numerical features, numerical labels).
        # So we just need to get the features relevant for the model.

        # To correctly get the features for a slice and predict,
        # you need to ensure the columns in `X_slice` match what the model was trained on.
        # This often involves dropping the label column, and possibly other columns
        # not used for training, and ensuring order is consistent.

        # Let's assume `data` is the preprocessed DataFrame where categorical features
        # have already been one-hot encoded, and `label_column` is already binarized.
        # The `categorical_feature` input here would then refer to the *original*
        # categorical feature, and you'd filter `sliced_data` based on that original column.
        # Then, you'd extract the encoded features for that slice.

        # A common approach: `data` itself is the pre-processed X.
        # Then `y` is separate. Let's adjust the signature to be more explicit
        # if `data` is the features `X`. If `data` is the combined X and y,
        # then we need to split it.

        # Given the prompt, "computes the performance metrics when the value of a given feature is held fixed. E.g., for education, it would print out the model metrics for data with a particular value for education."
        # This suggests `data` is the *original* dataset.
        # Therefore, we need `process_data` inside this function or ensure the input `data`
        # is sufficiently prepared for inference using `encoder` and `lb`.

        # Let's use a common pattern where `data` is the raw dataframe.
        # We need the `process_data` function. Add `from ml.data import process_data` at the top.
        from ml.data import process_data # Assuming this is available and handles encoding

        # The `process_data` function likely needs the full set of categorical features
        # that were used during training.
        # Let's assume `data` contains all columns used for training.

        # Identify categorical features from the original full dataset (before slicing)
        # This is a bit tricky without knowing the exact structure of your `data`
        # after initial processing. If `data` is the raw DataFrame,
        # we need to know the `categorical_features` list.
        # For simplicity, let's assume `data` is the raw input, and we get all its
        # object/category columns as `categorical_features` for `process_data`.
        
        # This part requires the `categorical_features` that were used during training
        # to be passed in or derived. For this example, let's assume `data` is the original
        # raw dataframe, and we extract `categorical_features` based on dtype.
        
        # First, ensure your `process_data` can handle single-slice data for inference.
        
        # Assuming `process_data` works as described in the Udacity project setup:
        # X_slice_processed, _, _, _ = process_data(
        #     sliced_data,
        #     categorical_features=[col for col in data.columns if data[col].dtype == 'object' and col != label_column],
        #     label=label_column,
        #     training=False,
        #     encoder=encoder,
        #     lb=lb
        # )
        # y_slice_true = lb.transform(y_slice.values).ravel()

        # Let's simplify for this implementation, assuming `data` is the already processed
        # numeric DataFrame, and `categorical_feature` corresponds to a column name in `data`
        # that resulted from one-hot encoding (e.g., 'workclass_Private'). This is less
        # aligned with the hint, but easier to implement directly within this function
        # without full knowledge of `process_data`'s internal workings on slices.

        # ********** REVISED ASSUMPTION FOR `performance_on_categorical_slice` **********
        # Given the hint, it's more likely `data` is the *raw* DataFrame.
        # We'll need to re-process `sliced_data` for each slice using the
        # provided `encoder` and `lb`. This implies `process_data` needs to be imported.

        # Re-import process_data if it's not already at the top of ml.model.py
        from ml.data import process_data

        # Get all categorical features from the original `data` for consistent processing.
        # You'll need to know which features were categorical during initial training.
        # For this example, let's assume `categorical_features` for `process_data`
        # are passed in or derived from the initial data processing step.
        # Let's assume the user passes a list of all `categorical_features`
        # to `performance_on_categorical_slice` if `data` is raw.
        # For now, I'll use a placeholder for `all_categorical_features_from_training`.
        # In a real scenario, this list would be known from your `data.py` setup.
        
        # If `data` is the raw DataFrame:
        
        # Step 1: Filter the raw data to get the slice
        raw_sliced_data = data[data[categorical_feature] == value]
        
        if raw_sliced_data.empty:
            continue

        # Step 2: Process the sliced raw data using the pre-trained encoder and lb
        # This requires knowing all categorical features that were used during initial training
        # to ensure correct column alignment for the model.
        # Assuming `data` passed to this function is the raw dataframe.
        # And assuming `process_data` takes the raw dataframe, a list of categorical features,
        # the label column, and the encoder/lb for inference.

        # You'd need a list of all categorical features from your dataset,
        # which `process_data` expects. Let's assume you have a `CAT_FEATURES` constant
        # or similar available from `ml.data` or defined elsewhere.
        # For the purpose of completing this, let's derive it or assume it's passed.
        # A robust solution would have this list passed in or saved with the model.
        
        # A simple way to get categorical features if not explicitly passed:
        all_categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if label_column in all_categorical_features:
            all_categorical_features.remove(label_column)


        X_slice_processed, y_slice_processed, _, _ = process_data(
            raw_sliced_data,
            categorical_features=all_categorical_features, # Use all original categorical features
            label=label_column,
            training=False,  # Important: Use training=False for inference
            encoder=encoder,
            lb=lb
        )
        
        # Make predictions on the processed slice
        preds = inference(model, X_slice_processed)
        
        # Compute metrics
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
    with open(path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object