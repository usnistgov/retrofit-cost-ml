# src/retrofit_cost_tool/main.py
"""
Train and save machine learning models for seismic retrofit cost prediction.
"""
import os
import json
import numpy as np
import warnings
from importlib import resources  # Replace pkg_resources
from .data_utils import load_data, preprocess_data, split_data
from .model_utils import train_ridge_model, train_elastic_net_model, train_random_forest_model, train_gradient_boosting_model, train_ols_model, train_glm_gamma_model, evaluate_model
from .model_io import save_model
from .model_selection import model_selection
from .plot_utils import plot_predictions

def main(verbose=True, random_state=42, save_models=False, save_metrics=False, suppress_warnings=True):
    if suppress_warnings:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    # Load data from package using modern approach
    with resources.path('retrofit_cost_tool.data', 'srce_train.csv') as data_path:
        data = load_data(str(data_path))


    # Preprocess data
    features_string = ['seismicity_pga050', 'p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'occup_cond', 'historic_dummy']
    features_num = ['area', 'bldg_age', 'stories']
    target = 'ystruct19'
    X, y = preprocess_data(data, features_string, features_num, target)

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = split_data(X, y, random_state=random_state)

    # Define model training functions and hyperparameter grids
    model_train_funcs = {
        'ols': (train_ols_model, ()),
        'glm_gamma': (train_glm_gamma_model, ()),
        'ridge': (train_ridge_model, (np.logspace(-3, 3, 100),)),
        'elastic_net': (train_elastic_net_model, (np.logspace(-3, 3, 100), np.linspace(0.1, 0.9, 10))),
        'random_forest': (train_random_forest_model, ([100, 200, 300], [None, 5, 10])),
        'gradient_boosting': (train_gradient_boosting_model, ([100, 200, 300], [None, 5, 10]))
    }

    # Perform model selection
    best_model_name, best_model, model_metrics, best_models = model_selection(model_train_funcs, None, X_train, y_train, random_state=random_state, verbose=verbose)
    print(f'Best model: {best_model_name}')

    # Evaluate best model on validation set
    rmse, mae, mape = evaluate_model(best_model, X_valid, y_valid)
    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%')

    # Save models and metrics
    if save_models:
        # Save models to package models directory
        with resources.path('retrofit_cost_tool.models', '') as model_dir:
            model_dir = str(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            
            for model_name, model in best_models.items():
                model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
                save_model(model, model_path)
                
            # Create best_model symlink
            best_model_path = os.path.join(model_dir, f'{best_model_name}_model.pkl')
            best_model_alias_path = os.path.join(model_dir, 'best_model.pkl')
            
            if os.path.exists(best_model_alias_path):
                os.remove(best_model_alias_path)
            os.symlink(f'{best_model_name}_model.pkl', best_model_alias_path)

    if save_metrics:
        # Save metrics to package models directory
        with resources.path('retrofit_cost_tool.models', '') as model_dir:
            model_dir = str(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            
            for model_name in best_models.keys():
                metrics_path = os.path.join(model_dir, f'{model_name}_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(model_metrics[model_name], f)
                    
            best_model_metrics_path = os.path.join(model_dir, 'best_model_metrics.json')
            with open(best_model_metrics_path, 'w') as f:
                json.dump(model_metrics[best_model_name], f)

    return best_model_name, best_model, model_metrics, X_valid, y_valid

if __name__ == "__main__":
    main(verbose=True, save_models=True, save_metrics=True)
