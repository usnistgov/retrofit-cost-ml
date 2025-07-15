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
# # Retrofit cost estimation tool: Model training and selection

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))

from retrofit_cost_tool import main
from retrofit_cost_tool.model_utils import evaluate_model

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %%
best_model_name, best_model, model_metrics, X_valid, y_valid = main(verbose=True, save_models=False, save_metrics=False)

# %%
# Explore the best model metrics or performance on holdout set
print(f'Best model: {best_model_name}')
rmse, mae, mape = evaluate_model(best_model, X_valid, y_valid)
print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%')

# %%

# %%
