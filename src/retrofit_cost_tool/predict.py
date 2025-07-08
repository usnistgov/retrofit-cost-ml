# src/retrofit_cost_tool/predict.py
"""
Seismic Retrofit Cost Predictor

This script allows users to load their own data and make predictions using pre-trained models.

Usage:
    python predict.py
"""

def predict(file_path=None, model_name='ridge'):
    if file_path is None:
        file_path = input("Enter the path to your data file: ")
    
    # Load user data
    data = load_data(file_path)
    
    # Preprocess data
    features_string = ['seismicity_pga050', 'p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'occup_cond', 'historic_dummy']
    features_num = ['area', 'bldg_age', 'stories']
    X, _ = preprocess_data(data, features_string, features_num, target=None)
    
    # Load the trained model
    model_path = os.path.join('..', '..', '..', 'models', f'{model_name}_model.pkl')
    model = load_model(model_path)
    
    # Make predictions
    predictions = model.predict(X)
    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    predict()
