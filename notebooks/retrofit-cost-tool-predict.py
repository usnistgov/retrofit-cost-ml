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
# # Retrofit cost tool: predictions with user data

# %%
import sys
import os
from retrofit_cost_tool import load_data, preprocess_data, predict, plot_predictions
import ipywidgets as widgets
from IPython.display import display
import pandas as pd

# %%
# Create a file uploader widget
file_uploader = widgets.FileUpload(
    description='Upload data file',
    accept='.csv'
)

# %%
# Create a dropdown widget for model selection
model_selector = widgets.Dropdown(
    options=['ridge', 'elastic_net', 'random_forest', 'gradient_boosting', 'ols', 'glm_gamma', 'best_model'],
    value='best_model',
    description='Select model:',
    disabled=False
)

# %%
# Create a button widget to trigger prediction
predict_button = widgets.Button(description='Make Predictions')

# Create a button widget to trigger plotting
plot_button = widgets.Button(description='Plot Predictions')

# Create a checkbox widget to save plots
save_plots_checkbox = widgets.Checkbox(value=False, description='Save plots')

predictions_and_actuals = None

# %%
output = widgets.Output()


# %%
# In your notebook, replace the predict call with:
def on_button_click(b):
    with output:
        output.clear_output(wait=True)
        global predictions_and_actuals
        file_name = file_uploader.value[0]['name']
        file_content = file_uploader.value[0]['content']
        with open(file_name, 'wb') as f:
            f.write(file_content.tobytes())
        
        # Load and predict using the updated functions
        data = load_data(file_name)
        target = list(data.columns)[-1]  # assume target is the last column
        
        # Use the predict function from the module
        predictions = predict(data, model_name=model_selector.value)
        actual_values = data[target]
        predictions_df = pd.DataFrame({'Predicted': predictions, 'Actual': actual_values})
        predictions_and_actuals = (predictions_df, actual_values)
        print(predictions_df.head())
        print("\nSummary Statistics:")
        print(predictions_df['Predicted'].describe())


# %%

# %%
def on_button_click(b):
    with output:
        output.clear_output(wait=True)
        #print("Make predictions button clicked")
        global predictions_and_actuals
        file_name = file_uploader.value[0]['name']
        file_content = file_uploader.value[0]['content']
        with open(file_name, 'wb') as f:
            f.write(file_content.tobytes())
        data = load_data(file_name)
        target = list(data.columns)[-1]  # assume target is the last column
        predictions = predict(data, model_name=model_selector.value)
        actual_values = data[target]
        predictions_df = pd.DataFrame({'Predicted': predictions, 'Actual': actual_values})
        predictions_and_actuals = (predictions_df, actual_values)
        print(predictions_df.head())
        print("\nSummary Statistics:")
        print(predictions_df['Predicted'].describe())



# %%
def on_plot_button_click(b):
    global predictions_and_actuals
    if predictions_and_actuals is not None:
        predictions_df, actual_values = predictions_and_actuals
        with output:
            #output.clear_output(wait=True)
            plot_predictions(predictions_df, actual_values, save_plots=False)
    else:
        with output:
            output.clear_output(wait=True)
            print("Please make predictions first")


# %%
predict_button.on_click(on_button_click)
plot_button.on_click(on_plot_button_click)

# %%
display(file_uploader)
display(model_selector)
display(predict_button)
display(output)

# %%
display(plot_button)
display(save_plots_checkbox)

# %%
