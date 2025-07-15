import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# Add the parent directory to the path to allow imports from ml/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, \
                     performance_on_categorical_slice, save_model, load_model

# Define paths for saving/loading models
MODEL_PATH = "model/model.pkl"
ENCODER_PATH = "model/encoder.pkl"
LB_PATH = "model/lb.pkl"
SLICE_OUTPUT_PATH = "slice_output.txt"

def run_pipeline():
    # Load the census.csv data.
    try:
        data = pd.read_csv("data/census.csv")
    except FileNotFoundError:
        print("Error: census.csv not found. Make sure it's in the 'data/' directory.")
        return

    # Define categorical features and label
    # These should match the features you expect in your dataset
    # and what process_data is designed to handle.
    CAT_FEATURES = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    ]
    LABEL_COLUMN = 'salary'

    # Clean column names (replace hyphens with underscores)
    data.columns = data.columns.str.replace('-', '_')
    LABEL_COLUMN = LABEL_COLUMN.replace('-', '_')
    CAT_FEATURES = [f.replace('-', '_') for f in CAT_FEATURES]


    # Split the data into a training dataset and a test dataset.
    train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data[LABEL_COLUMN])

    # Process both the test and the training data.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL_COLUMN, training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label=LABEL_COLUMN, training=False, encoder=encoder, lb=lb
    )

    # Train the model on the training dataset.
    model = train_model(X_train, y_train)

    # Create 'model' directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Save the model and the encoder.
    save_model(model, MODEL_PATH)
    save_model(encoder, ENCODER_PATH)
    save_model(lb, LB_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")
    print(f"LabelBinarizer saved to {LB_PATH}")

    # Run the model inferences on the test dataset.
    preds = inference(model, X_test)

    # Compute overall performance metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print("\n--- Overall Model Performance ---")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

    # Load the model and encoder for consistency, though they are already in memory.
    # This step is more for demonstrating the load functionality.
    # loaded_model = load_model(MODEL_PATH)
    # loaded_encoder = load_model(ENCODER_PATH)
    # loaded_lb = load_model(LB_PATH)
    # print("\nModel and encoders successfully loaded for verification.")

    # Computes performance on data slices using the performance_on_categorical_slice function.
    # And save the output to slice_output.txt.
    print(f"\nComputing performance on data slices and saving to {SLICE_OUTPUT_PATH}...")
    with open(SLICE_OUTPUT_PATH, 'w') as f:
        f.write("--- Performance on Categorical Slices ---\n\n")
        for feature in CAT_FEATURES:
            # For `performance_on_categorical_slice`, we pass the original `test` DataFrame
            # so that it can correctly slice based on the raw categorical values.
            # It will then internally process these slices using the provided encoder and lb.
            slice_metrics = performance_on_categorical_slice(
                model, test, feature, LABEL_COLUMN, encoder, lb
            )
            f.write(f"Feature: {feature}\n")
            for slice_name, metrics in slice_metrics.items():
                # Extract the original value from the slice_name (e.g., 'workclass_Private' -> 'Private')
                # This handles cases where the key might be just the value if no prefix was added
                original_value = slice_name.split('_', 1)[1] if '_' in slice_name and slice_name.startswith(feature + '_') else slice_name
                
                # Count instances of this original value in the raw test set
                # Important: Count from the original `test` DataFrame before processing
                count = test[test[feature] == original_value].shape[0]

                f.write(f"  {original_value} (Count: {count})\n")
                f.write(f"    Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['fbeta']:.4f}\n")
            f.write("\n")
    print(f"Slice performance report saved to {SLICE_OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()