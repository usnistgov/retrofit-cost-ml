# model_io.py

import joblib

def save_model(model, file_path):
    """Save a trained model to a file."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load a saved model from a file."""
    return joblib.load(file_path)
