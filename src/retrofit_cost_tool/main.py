# src/retrofit_cost_tool/main.py
"""
Train and save machine learning models for seismic retrofit cost prediction.
"""
import numpy as np
import os
from .data_utils import load_data, preprocess_data, split_data
from .model_utils import train_ridge_model, train_elastic_net_model, train_random_forest_model, train_gradient_boosting_model, train_ols_model, train_glm_gamma_model, evaluate_model
from .model_io import save_model
from .model_selection import model_selection

def main(verbose=True, random_state=42):
    # Load data
    file_path = os.path.join('..', '..', '..', 'data', 'srce_train.csv')
    data = load_data(file_path)

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
    best_model_name, best_model, model_metrics = model_selection(model_train_funcs, None, X_train, y_train, random_state=random_state, verbose=verbose)
    print(f'Best model: {best_model_name}')

    # Evaluate best model on validation set
    rmse, mae, mape = evaluate_model(best_model, X_valid, y_valid)
    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%')

    # Save models and metrics
    model_dir = os.path.join('..', '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    for model_name in model_train_funcs.keys():
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        model = model_train_funcs[model_name][0](X, y, *model_train_funcs[model_name][1])
        save_model(model, model_path)
        metrics_path = os.path.join(model_dir, f'{model_name}_metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(model_metrics[model_name], f)

    # Save best model metrics
    best_model_metrics_path = os.path.join(model_dir, 'best_model_metrics.json')
    with open(best_model_metrics_path, 'w') as f:
        import json
        json.dump(model_metrics[best_model_name], f)

    best_model_path = os.path.join(model_dir, f'{best_model_name}_model.pkl')
    best_model_alias_path = os.path.join(model_dir, 'best_model.pkl')
    if os.path.exists(best_model_alias_path):
        os.remove(best_model_alias_path)
    import os
    os.symlink(os.path.basename(best_model_path), best_model_alias_path)

if __name__ == "__main__":
    main(verbose=True)
