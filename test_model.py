import pytest
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

# Define the path to your saved model
MODEL_PATH = 'social_media_addiction_model.joblib'

@pytest.fixture(scope="module")
def trained_model():
    """Fixture to load the model once for all tests."""
    model = joblib.load(MODEL_PATH)
    return model

@pytest.fixture(scope="module")
def sample_input():
    """Fixture to provide a consistent sample input for inference tests."""
    # This input must match the structure of X after dropping object columns
    # Columns: age, daily_usage_hours, num_platforms_used, avg_session_minutes, night_usage, mental_health_score, screen_time_before_sleep
    return pd.DataFrame([[20, 3.0, 3, 25.0, 0, 50, 0]],
                        columns=['age', 'daily_usage_hours', 'num_platforms_used', 'avg_session_minutes', 'night_usage', 'mental_health_score', 'screen_time_before_sleep'])

def test_model_loading(trained_model):
    """Test that the model loads correctly and is a LightGBM classifier."""
    assert isinstance(trained_model, lgb.LGBMClassifier)

def test_inference_shape(trained_model, sample_input):
    """Test that the model's prediction output has the expected shape."""
    predictions = trained_model.predict(sample_input)
    assert predictions.shape == (1,) # Expect a single prediction for one input

def test_known_inference_result(trained_model, sample_input):
    """Test that a known input produces an expected prediction.
    
    Based on the previous training, a 'typical' input might result in 'Medium'
    addiction level. Given the label mapping {'High': 0, 'Low': 1, 'Medium': 2},
    we expect an encoded value of 2 for 'Medium'.
    You might need to adjust the expected_prediction if your model's behavior or
    the sample input changes.
    """
    predictions = trained_model.predict(sample_input)
    expected_prediction = 2 # Assuming 'Medium' is the most likely output for this generic input
    assert predictions[0] == expected_prediction
