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
#     display_name: Python [conda env:test-retrofits]
#     language: python
#     name: conda-env-test-retrofits-py
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
# Test prediction with packaged data
from importlib import resources

# Load the packaged synthetic data directly
with resources.path('retrofit_cost_tool.data', 'synthetic_data.csv') as data_path:
    data = load_data(str(data_path))

# Make predictions using the predict function
predictions = predict(data, model_name='best_model')

if predictions is not None:
    print(f"✅ Generated {len(predictions)} predictions using packaged data")
    print(f"Sample predictions: {predictions[:5]}")
else:
    print("❌ Prediction failed")

# %%
predictions_df = pd.DataFrame({'Predicted': predictions, 'Actual': data['ystruct19']})
print("✅ Predictions from packaged data:")
print(predictions_df.head())

# %%
# Create widgets
file_uploader = widgets.FileUpload(
    description='Upload data file',
    accept='.csv'
)

model_selector = widgets.Dropdown(
    options=['ridge_model', 'elastic_net_model', 'random_forest_model', 
             'gradient_boosting_model', 'ols_model', 'glm_gamma_model', 'best_model'],
    value='best_model',
    description='Select model:',
)

predict_button = widgets.Button(description='Make Predictions')
plot_button = widgets.Button(description='Plot Predictions')
save_plots_checkbox = widgets.Checkbox(value=False, description='Save plots')

# Global variable to store predictions
predictions_and_actuals = None

output = widgets.Output()

def on_button_click(b):
    global predictions_and_actuals  # Add this line
    with output:
        output.clear_output(wait=True)
        if file_uploader.value:
            # Process uploaded file
            file_name = file_uploader.value[0]['name']
            file_content = file_uploader.value[0]['content']
            with open(file_name, 'wb') as f:
                f.write(file_content.tobytes())
            
            data = load_data(file_name)
            target = list(data.columns)[-1]
            predictions = predict(data, model_name=model_selector.value)
            actual_values = data[target]
            
            predictions_df = pd.DataFrame({'Predicted': predictions, 'Actual': actual_values})
            predictions_and_actuals = (predictions_df, actual_values)  # Set the global variable
            
            print("✅ Predictions from uploaded file:")
            print(predictions_df.head())
            print("\nSummary Statistics:")
            print(predictions_df['Predicted'].describe())
        else:
            print("Please upload a CSV file first")

def on_plot_button_click(b):
    global predictions_and_actuals
    if predictions_and_actuals is not None:
        predictions_df, actual_values = predictions_and_actuals
        with output:
            try:
                # Pass the DataFrame directly - it already has 'Predicted' and 'Actual' columns
                plot_predictions(predictions_df, actual_values, save_plots=save_plots_checkbox.value)
            except Exception as e:
                print(f"Plotting error: {e}")
                # Try with explicit column access
                try:
                    plot_predictions(predictions_df['Predicted'], predictions_df['Actual'], save_plots=save_plots_checkbox.value)
                except Exception as e2:
                    print(f"Alternative plotting also failed: {e2}")
                    import traceback
                    traceback.print_exc()
    else:
        with output:
            print("Please make predictions first")


predict_button.on_click(on_button_click)
plot_button.on_click(on_plot_button_click)

# %%
# Display widgets
display(file_uploader)
display(model_selector)
display(predict_button)
display(output)
display(plot_button)
display(save_plots_checkbox)
