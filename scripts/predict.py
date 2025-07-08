# predict.py
"""
Seismic Retrofit Cost Predictor

This script allows users to load their own data and make predictions using pre-trained models.

Usage:
    python predict.py
"""

import os
from seismic_retrofit_cost_predictor.data_utils import load_data, preprocess_data
from seismic_retrofit_cost_predictor.model_io import load_model

def main():
    # Specify the features used during training
    features_string = ['seismicity_pga050', 'p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'occup_cond', 'historic_dummy']
    features_num = ['area', 'bldg_age', 'stories']
    
    # Load user data
    while True:
        file_path = input("Enter the path to your data file: ")
        if os.path.exists(file_path):
            try:
                X_user = load_data(file_path)
                X_user, _ = preprocess_data(X_user, features_string, features_num, target=None)
                break
            except Exception as e:
                print(f"Error loading or preprocessing data: {e}")
        else:
            print("Invalid file path. Please try again.")
    
    # Load the trained model
    while True:
        model_name = input("Enter the name of the model to use (ridge, elastic_net, random_forest, gradient_boosting): ")
        if model_name in ['ridge', 'elastic_net', 'random_forest', 'gradient_boosting']:
            model_path = f'models/{model_name}_model.pkl'
            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                    break
                except Exception as e:
                    print(f"Error loading model: {e}")
            else:
                print("Model not found. Please try again.")
        else:
            print("Invalid model name. Please try again.")
    
    # Make predictions
    try:
        predictions = model.predict(X_user)
        print("Predictions:")
        print(predictions)
    except Exception as e:
        print(f"Error making predictions: {e}")

if __name__ == "__main__":
    main()
