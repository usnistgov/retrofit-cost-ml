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
from datetime import datetime

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
# Check if ground truth exists in packaged data
has_ground_truth = 'ystruct19' in data.columns
if has_ground_truth:
    predictions_df = pd.DataFrame({'Predicted': predictions, 'Actual': data['ystruct19']})
    print("✅ Predictions from packaged data (with ground truth):")
else:
    predictions_df = pd.DataFrame({'Predicted': predictions})
    print("✅ Predictions from packaged data (no ground truth):")
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

predict_button = widgets.Button(description='Make Predictions', button_style='primary')
plot_button = widgets.Button(description='Plot Predictions', button_style='info')
save_csv_button = widgets.Button(description='Save to CSV', button_style='success')

save_plots_checkbox = widgets.Checkbox(value=False, description='Save plots')

# Global variables to store predictions and metadata
predictions_and_actuals = None
current_data_info = None

output = widgets.Output()

def on_button_click(b):
    global predictions_and_actuals, current_data_info
    with output:
        output.clear_output(wait=True)
        if file_uploader.value:
            # Process uploaded file
            file_name = file_uploader.value[0]['name']
            file_content = file_uploader.value[0]['content']
            with open(file_name, 'wb') as f:
                f.write(file_content.tobytes())
            
            data = load_data(file_name)
            
            # Check if ground truth exists
            target_column = 'ystruct19'  # Expected target column name
            has_actual = target_column in data.columns
            
            predictions = predict(data, model_name=model_selector.value)
            
            if has_actual:
                actual_values = data[target_column]
                predictions_df = pd.DataFrame({
                    'Predicted': predictions, 
                    'Actual': actual_values
                })
                predictions_and_actuals = (predictions_df, actual_values, True)  # True = has ground truth
                
                print("✅ Predictions from uploaded file (with ground truth):")
                print(predictions_df.head())
                print("\nSummary Statistics:")
                print(predictions_df.describe())
                
                # Calculate accuracy metrics
                mae = abs(predictions_df['Predicted'] - predictions_df['Actual']).mean()
                rmse = ((predictions_df['Predicted'] - predictions_df['Actual'])**2).mean()**0.5
                print(f"\nAccuracy Metrics:")
                print(f"MAE: ${mae:,.2f}")
                print(f"RMSE: ${rmse:,.2f}")
                
            else:
                predictions_df = pd.DataFrame({'Predicted': predictions})
                predictions_and_actuals = (predictions_df, None, False)  # False = no ground truth
                
                print("✅ Predictions from uploaded file (no ground truth available):")
                print(predictions_df.head())
                print("\nSummary Statistics:")
                print(predictions_df['Predicted'].describe())
            
            # Store metadata for CSV export
            current_data_info = {
                'filename': file_name,
                'model_used': model_selector.value,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'has_ground_truth': has_actual,
                'num_predictions': len(predictions)
            }
            
        else:
            print("Please upload a CSV file first")

def on_plot_button_click(b):
    global predictions_and_actuals
    if predictions_and_actuals is not None:
        predictions_df, actual_values, has_ground_truth = predictions_and_actuals
        with output:
            try:
                if has_ground_truth:
                    # Plot with ground truth comparison
                    plot_predictions(predictions_df, actual_values, save_plots=save_plots_checkbox.value)
                else:
                    # Plot predictions only (histogram/distribution)
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Histogram of predictions
                    ax1.hist(predictions_df['Predicted'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax1.set_xlabel('Predicted Retrofit Cost ($)')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Distribution of Predicted Costs')
                    ax1.ticklabel_format(style='plain', axis='x')
                    
                    # Box plot
                    ax2.boxplot(predictions_df['Predicted'])
                    ax2.set_ylabel('Predicted Retrofit Cost ($)')
                    ax2.set_title('Prediction Distribution (Box Plot)')
                    ax2.ticklabel_format(style='plain', axis='y')
                    
                    plt.tight_layout()
                    
                    if save_plots_checkbox.value:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        plt.savefig(f'predictions_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
                        print("📊 Plot saved as PNG file")
                    
                    plt.show()
                    
            except Exception as e:
                print(f"Plotting error: {e}")
                import traceback
                traceback.print_exc()
    else:
        with output:
            print("Please make predictions first")

def on_save_csv_click(b):
    global predictions_and_actuals, current_data_info
    if predictions_and_actuals is not None:
        predictions_df, _, has_ground_truth = predictions_and_actuals
        
        with output:
            try:
                # Create filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_name = current_data_info['filename'].replace('.csv', '') if current_data_info else 'predictions'
                csv_filename = f"{base_name}_predictions_{timestamp}.csv"
                
                # Add metadata as comments (if supported) or as additional columns
                export_df = predictions_df.copy()
                
                # Add metadata columns
                if current_data_info:
                    export_df['model_used'] = current_data_info['model_used']
                    export_df['prediction_timestamp'] = current_data_info['timestamp']
                    export_df['source_file'] = current_data_info['filename']
                
                # Save to CSV
                export_df.to_csv(csv_filename, index=False)
                
                print(f"✅ Predictions saved to: {csv_filename}")
                print(f"📊 Exported {len(export_df)} predictions")
                
                if has_ground_truth:
                    print("📋 Columns: Predicted, Actual, model_used, prediction_timestamp, source_file")
                else:
                    print("📋 Columns: Predicted, model_used, prediction_timestamp, source_file")
                    
            except Exception as e:
                print(f"❌ Error saving CSV: {e}")
    else:
        with output:
            print("Please make predictions first")

predict_button.on_click(on_button_click)
plot_button.on_click(on_plot_button_click)
save_csv_button.on_click(on_save_csv_click)

# %%
# Display widgets in organized layout
upload_section = widgets.VBox([
    widgets.HTML("<h3>📁 Data Upload</h3>"),
    file_uploader,
    model_selector
])

action_section = widgets.VBox([
    widgets.HTML("<h3>🔧 Actions</h3>"),
    widgets.HBox([predict_button, plot_button, save_csv_button]),
    save_plots_checkbox
])

results_section = widgets.VBox([
    widgets.HTML("<h3>📊 Results</h3>"),
    output
])

# Display all sections
display(upload_section)
display(action_section) 
display(results_section)

# %%
# Optional: Instructions for users
instructions = widgets.HTML("""
<div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0;">
<h4>📋 Instructions:</h4>
<ol>
<li><strong>Upload your CSV file</strong> with building data (area, bldg_age, stories, etc.)</li>
<li><strong>Select a model</strong> (start with 'best_model')</li>
<li><strong>Make Predictions</strong> to generate retrofit cost estimates</li>
<li><strong>Plot Predictions</strong> to visualize results</li>
<li><strong>Save to CSV</strong> to export predictions with metadata</li>
</ol>
<p><strong>Note:</strong> Ground truth values (ystruct19) are optional. The tool works with prediction-only data.</p>
</div>
""")

display(instructions)
