# src/retrofit_cost_tool/predict.py
"""
Seismic Retrofit Cost Predictor
This script allows users to load their own data and make predictions using pre-trained models.
"""
import os
from pathlib import Path
from .data_utils import load_data, preprocess_data
from .model_io import load_model

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def predict(data, model_name='best_model'):
    """Make predictions using a trained model."""
    # Preprocess data
    features_string = ['seismicity_pga050', 'p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'occup_cond', 'historic_dummy']
    features_num = ['area', 'bldg_age', 'stories']
    X, _ = preprocess_data(data, features_string, features_num, target=None)

    # Load the trained model
    project_root = get_project_root()
    model_path = project_root / 'models' / f'{model_name}.pkl'
    model = load_model(str(model_path))

    # Make predictions
    predictions = model.predict(X)
    return predictions

def main():
    """Entry point for command-line usage"""
    # Load default data for prediction
    project_root = get_project_root()
    data_path = project_root / 'data' / 'srce_all.csv'

    print(f"Loading data from: {data_path}")
    data = load_data(str(data_path))

    print("Making predictions...")
    predictions = predict(data)

    print("Predictions:")
    print(predictions)
    return predictions

if __name__ == "__main__":
    main()
