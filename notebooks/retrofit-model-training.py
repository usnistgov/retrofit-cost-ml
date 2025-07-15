# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:retrofits]
#     language: python
#     name: conda-env-retrofits-py
# ---

# %% [markdown]
# # Notebook documenting model training, selection, and saving pre-trained models

# %%
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from retrofit_cost_tool import load_data, preprocess_data, split_data, evaluate_model, save_model, model_selection
from retrofit_cost_tool import train_ridge_model, train_elastic_net_model, train_random_forest_model, train_gradient_boosting_model, train_ols_model, train_glm_gamma_model
from retrofit_cost_tool.main import main

# %%
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %%
# Load training data
file_path = '../data/srce_train.csv'
data = load_data(file_path)

# %%
# Preprocess data
features_string = ['seismicity_pga050', 'p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'occup_cond', 'historic_dummy']
features_num = ['area', 'bldg_age', 'stories']
target = 'ystruct19'
X, y = preprocess_data(data, features_string, features_num, target)

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = split_data(X, y)


# %%
# Train models
alpha_grid = np.logspace(-3, 3, 100)
l1_ratio_grid = np.linspace(0.1, 0.9, 10)
n_estimators_grid = [100, 200, 300]
max_depth_grid = [None, 5, 10]

model_train_funcs = {
    'ols': (train_ols_model, ()),
    'glm_gamma': (train_glm_gamma_model, ()),
    'ridge': (train_ridge_model, (alpha_grid,)),
    'elastic_net': (train_elastic_net_model, (alpha_grid, l1_ratio_grid)),
    'random_forest': (train_random_forest_model, (n_estimators_grid, max_depth_grid)),
    'gradient_boosting': (train_gradient_boosting_model, (n_estimators_grid, max_depth_grid))
}

# Model selection
best_model_name, best_model, model_metrics, best_models = model_selection(model_train_funcs, None, X_train, y_train, verbose=True)
print(f'Best model: {best_model_name}')


# %%
# Evaluate best model
rmse, mae, mape = evaluate_model(best_model, X_valid, y_valid)
print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%')


# %%
# Save best model
model_path = os.path.join('../models', f'{best_model_name}_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
save_model(best_model, model_path)
print(f'Saved {best_model_name} model to {model_path}')

# %%
best_model_alias_path = os.path.join('..', 'models', 'best_model.pkl')

# Create a symbolic link to the best model
if os.path.exists(best_model_alias_path):
    os.remove(best_model_alias_path)
os.symlink(os.path.basename(model_path), best_model_alias_path)

print(f'Created alias for best model: {best_model_alias_path}')

# %%
# Save all models
for model_name, model in best_models.items():
    model_path = os.path.join('../models', f'{model_name}_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(model, model_path)
    print(f'Saved {model_name} model to {model_path}')
