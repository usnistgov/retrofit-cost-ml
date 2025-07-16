# src/retrofit_cost_tool/predict.py
"""
Seismic Retrofit Cost Predictor
This script allows users to load their own data and make predictions using pre-trained models.
"""
import os
import sys
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
    # Parse command line arguments
    if len(sys.argv) > 1:
        # User provided their own data file
        data_file = sys.argv[1]
        if not os.path.exists(data_file):
            print(f"Error: Data file '{data_file}' not found.")
            return None
        print(f"Loading user data from: {data_file}")
        data_path = data_file
    else:
        # Use default synthetic data
        # Load default data for prediction
        project_root = get_project_root()
        data_path = project_root / 'data' / 'synthetic_data.csv'
        print(f"Loading default synthetic data from: {data_path}")

    # Optional model name argument
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'best_model'

    try:
        data = load_data(data_path)
        print(f"Making predictions using model: {model_name}")
        predictions = predict(data, model_name)

        print("Predictions:")
        print(predictions)
        return predictions
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


if __name__ == "__main__":
    main()
