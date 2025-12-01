import pytest # noqa: F401
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference

# Fixture to create a small sample dataset for testing
@pytest.fixture
def data():
    """
    Creates a simple dataframe for testing.
    """
    df = pd.DataFrame({
        'age': [39, 50, 38],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private'],
        'fnlgt': [77516, 83311, 215646],
        'education': ['Bachelors', 'Bachelors', 'HS-grad'],
        'education-num': [13, 13, 9],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family'],
        'race': ['White', 'White', 'White'],
        'sex': ['Male', 'Male', 'Male'],
        'capital-gain': [2174, 0, 0],
        'capital-loss': [0, 0, 0],
        'hours-per-week': [40, 13, 40],
        'native-country': ['United-States', 'United-States', 'United-States'],
        'salary': ['<=50K', '<=50K', '<=50K']
    })
    return df

def test_process_data_shape(data):
    """
    Test 1: Check if process_data returns the correct number of rows.
    """
    cat_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    
    # Process the data
    X, y, _, _ = process_data(
        data, 
        categorical_features=cat_features, 
        label='salary', 
        training=True
    )
    
    # Assertions
    assert X.shape[0] == data.shape[0]
    assert len(y) == data.shape[0]

def test_train_model_type(data):
    """
    Test 2: Check if train_model returns a RandomForestClassifier.
    """
    cat_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    
    X, y, _, _ = process_data(
        data, 
        categorical_features=cat_features, 
        label='salary', 
        training=True
    )
    
    model = train_model(X, y)
    
    # Assertion
    assert isinstance(model, RandomForestClassifier)

def test_inference_shape(data):
    """
    Test 3: Check if inference returns predictions for every row.
    """
    cat_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    
    X, y, _, _ = process_data(
        data, 
        categorical_features=cat_features, 
        label='salary', 
        training=True
    )
    model = train_model(X, y)
    
    preds = inference(model, X)
    
    # Assertion
    assert len(preds) == len(X)
