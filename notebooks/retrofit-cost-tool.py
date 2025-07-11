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
# # Notebook illustrating retrofit cost estimation with predictive models

# %%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
from retrofit_cost_tool import (load_data, preprocess_data, load_model,
                                 train_ridge_model, train_elastic_net_model,
                                 train_random_forest_model, train_gradient_boosting_model,
                                 train_ols_model, train_glm_gamma_model,
                                 model_selection)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import numpy as np

# %%
# Specify the features used during training
features_string = ['seismicity_pga050', 'p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'occup_cond', 'historic_dummy']
features_num = ['area', 'bldg_age', 'stories']

# Create a file uploader widget
file_uploader = widgets.FileUpload(
    description='Upload data file',
    accept='.csv'
)
# Create a dictionary of model training functions
model_train_funcs = {
    'ols': (train_ols_model, ()),
    'glm_gamma': (train_glm_gamma_model, ()),
    'ridge': (train_ridge_model, (alpha_grid,)),
    'elastic_net': (train_elastic_net_model, (alpha_grid, l1_ratio_grid)),
    'random_forest': (train_random_forest_model, (n_estimators_grid, max_depth_grid)),
    'gradient_boosting': (train_gradient_boosting_model, (n_estimators_grid, max_depth_grid))
}

# Create a dropdown widget for model selection
model_selector = widgets.Dropdown(
    options=list(model_train_funcs.keys()),
    value='ridge',
    description='Select model:',
    disabled=False
)

# Create a button widget to trigger prediction
predict_button = widgets.Button(description='Make Predictions')

# Define a function to load data and make predictions
def make_predictions(file_uploader, model_selector):
    """
    Load user data, preprocess it, load a trained model, and make predictions.

    Args:
    - file_uploader (widgets.FileUpload): File uploader widget.
    - model_selector (widgets.Dropdown): Dropdown widget for model selection.

    Returns:
    - None
    """
    # Load user data
    file_path = list(file_uploader.value.keys())[0]
    with open(file_path, 'wb') as f:
        f.write(file_uploader.value[file_path]['content'])

    X_user = load_data(file_path)
    X_user, _ = preprocess_data(X_user, features_string, features_num, target=None)

    # Load the trained model
    model_name = model_selector.value
    model_path = os.path.join('..', 'models', f'{model_name}_model.pkl')
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(X_user)
    print("Predictions:")
    print(predictions)

# Define a function to handle button click
def on_button_click(b):
    make_predictions(file_uploader, model_selector)

# Link the button click to the on_button_click function
predict_button.on_click(on_button_click)


# %%
# Display the widgets
display(file_uploader)
display(model_selector)
display(predict_button)

# %%
# Example usage
print("Example usage:")
print("Load synthetic data...")
synthetic_data_path = os.path.join('..', 'data', 'synthetic_data.csv')
synthetic_data = load_data(synthetic_data_path)

# %%
print(synthetic_data.head())

# %%
print("Preprocess...")
target = 'ystruct19'
X_synthetic, _ = preprocess_data(synthetic_data, features_string, features_num, target=target)
model_name = 'ridge'
model_path = os.path.join('..', 'models', f'{model_name}_model.pkl')
model = load_model(model_path)
predictions = model.predict(X_synthetic)
print("Predictions using synthetic data:")
print(predictions)

# %%
# Compare predicted and actual values
actual_values = synthetic_data[target]
predictions_df = pd.DataFrame({'Predicted': predictions, 'Actual': actual_values})
print("Predicted vs Actual Values:")
print(predictions_df)


# %%
# Plot predicted vs actual values
def plot_predictions(predictions_df, actual_values, plot_scatter=True, plot_histograms=True, save_plots=False):
    """
    Plot predicted vs actual values.

    Args:
    - predictions_df (pd.DataFrame): DataFrame containing predicted and actual values.
    - actual_values (array-like): Actual values.
    - plot_scatter (bool, optional): Whether to plot a scatter plot. Defaults to True.
    - plot_histograms (bool, optional): Whether to plot histograms. Defaults to True.
    - save_plots (bool, optional): Whether to save the plots. Defaults to False.

    Returns:
    - None
    """
    if plot_histograms:
        sns.histplot(predictions_df['Predicted'], kde=True, label='Predicted Costs')
        sns.histplot(actual_values, kde=True, label='Actual Costs', color='red')
        plt.legend()
        plt.title('Cost Predictions')
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        if save_plots:
            plt.savefig('cost_predictions.png', bbox_inches='tight', dpi=300)
        plt.show()

    if plot_scatter:
        plt.scatter(actual_values, predictions_df['Predicted'])
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        if save_plots:
            plt.savefig('predicted_vs_actual.png', bbox_inches='tight', dpi=300)
        plt.show()


# %%
# Plot predicted vs actual values
# NB: DEPRECATED
plt.scatter(actual_values, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.savefig('predicted_vs_actual.png', bbox_inches='tight', dpi=300)
plt.show()


# %%
# for synthetic user data with actual values, plot actual vs predicted

# Call the function with save_plots=True
plot_predictions(predictions_df, actual_values, save_plots=True)

# %%

# %%
