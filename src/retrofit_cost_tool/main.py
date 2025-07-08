# src/retrofit_cost_tool/main.py
"""
Train and save machine learning models for seismic retrofit cost prediction.
"""
import numpy as np

def main():
    # Load data
    file_path = os.path.join('..', '..', '..', 'data', 'srce_train.csv')
    data = load_data(file_path)
    
    # Preprocess data
    features_string = ['seismicity_pga050', 'p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'occup_cond', 'historic_dummy']
    features_num = ['area', 'bldg_age', 'stories']
    target = 'ystruct19'
    X, y = preprocess_data(data, features_string, features_num, target)
    
    # Split data
    X_train, X_valid, y_train, y_valid = split_data(X, y)
    
    # Train models
    alpha_grid = np.logspace(-3, 3, 100)
    l1_ratio_grid = np.linspace(0.1, 0.9, 10)
    n_estimators_grid = [100, 200, 300]
    max_depth_grid = [None, 5, 10]
    models = {
        'ridge': train_ridge_model(X_train, y_train, alpha_grid),
        'elastic_net': train_elastic_net_model(X_train, y_train, alpha_grid, l1_ratio_grid),
        'random_forest': train_random_forest_model(X_train, y_train, n_estimators_grid, max_depth_grid),
        'gradient_boosting': train_gradient_boosting_model(X_train, y_train, n_estimators_grid, max_depth_grid)
    }
    
    # Evaluate models
    for model_name, model in models.items():
        rmse = evaluate_model(model, X_valid, y_valid)
        print(f'{model_name.capitalize()} RMSE: {rmse:.4f}')
    
    # Save models
    model_dir = os.path.join('..', '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    for model_name, model in models.items():
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        save_model(model, model_path)

if __name__ == "__main__":
    main()
