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

# %% [markdown]
# # Demo user module

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

feature_selector = widgets.Dropdown(
    options=['p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'historic_dummy', 'occup_cond'],
    value='p_obj_dummy',
    description='Group by:',
)

# filename input widget
filename_input = widgets.Text(
    placeholder='Optional: custom filename (without .csv)',
    description='Output name:',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='300px')
)

predict_button = widgets.Button(description='Make Predictions', button_style='primary')
plot_button = widgets.Button(description='Plot Predictions', button_style='info')
save_csv_button = widgets.Button(description='Save to CSV', button_style='success')
summary_button = widgets.Button(description='Feature Summary', button_style='warning')

# Global variable to store uploaded data for summaries
uploaded_data = None


save_plots_checkbox = widgets.Checkbox(value=False, description='Save plots')

# Global variables to store predictions and metadata
predictions_and_actuals = None
current_data_info = None
uploaded_data = None  # Store original data for CSV export

output = widgets.Output()

def on_button_click(b):
    global predictions_and_actuals, current_data_info, uploaded_data  # Add uploaded_data here
    with output:
        output.clear_output(wait=True)
        if file_uploader.value:
            # Process uploaded file
            file_name = file_uploader.value[0]['name']
            file_content = file_uploader.value[0]['content']
            with open(file_name, 'wb') as f:
                f.write(file_content.tobytes())
            
            data = load_data(file_name)
            uploaded_data = data.copy() # store original user data
            
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


def on_summary_click(b):
    global predictions_and_actuals, uploaded_data
    if predictions_and_actuals is not None and uploaded_data is not None:
        predictions_df, _, has_ground_truth = predictions_and_actuals
        selected_feature = feature_selector.value
        
        with output:
            try:
                # Check if feature exists in data
                if selected_feature not in uploaded_data.columns:
                    print(f"❌ Feature '{selected_feature}' not found in your data")
                    return
                
                # Create summary DataFrame
                summary_data = uploaded_data.copy()
                summary_data['Predicted_Cost'] = predictions_df['Predicted']
                
                # Group by selected feature
                grouped = summary_data.groupby(selected_feature)['Predicted_Cost']
                
                print(f"📊 **Summary Statistics by {selected_feature}:**")
                print("=" * 50)
                
                summary_stats = grouped.agg(['count', 'mean', 'median', 'std', 'min', 'max'])
                
                # Format the output nicely
                for group_value, stats in summary_stats.iterrows():
                    print(f"\n🏢 {selected_feature} = {group_value}:")
                    print(f"   Count: {stats['count']:.0f} buildings")
                    print(f"   Mean Cost: ${stats['mean']:,.2f}")
                    print(f"   Median Cost: ${stats['median']:,.2f}")
                    print(f"   Std Dev: ${stats['std']:,.2f}")
                    print(f"   Range: ${stats['min']:,.2f} - ${stats['max']:,.2f}")
                
                # Add percentage breakdown
                print(f"\n📈 **Distribution by {selected_feature}:**")
                value_counts = uploaded_data[selected_feature].value_counts()
                for value, count in value_counts.items():
                    percentage = (count / len(uploaded_data)) * 100
                    avg_cost = grouped.get_group(value).mean()
                    print(f"   {selected_feature} = {value}: {count} buildings ({percentage:.1f}%) - Avg: ${avg_cost:,.2f}")
                
            except Exception as e:
                print(f"❌ Error generating summary: {e}")
                import traceback
                traceback.print_exc()
    else:
        with output:
            print("Please make predictions first")

            
def on_plot_button_click(b):
    global predictions_and_actuals
    if predictions_and_actuals is not None:
        predictions_df, actual_values, has_ground_truth = predictions_and_actuals
        with output:
            try:
                plot_predictions(
                    predictions_df, 
                    actual_values if has_ground_truth else None, 
                    plot_scatter=has_ground_truth, 
                    plot_histograms=True, 
                    save_plots=save_plots_checkbox.value
                )
                    
            except Exception as e:
                print(f"Plotting error: {e}")
                import traceback
                traceback.print_exc()
    else:
        with output:
            print("Please make predictions first")


def on_save_csv_click(b):
    global predictions_and_actuals, current_data_info, uploaded_data
    if predictions_and_actuals is not None:
        predictions_df, _, has_ground_truth = predictions_and_actuals
        
        with output:
            try:
                # Determine filename
                custom_name = filename_input.value.strip()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if custom_name:
                    # Use custom name with timestamp
                    csv_filename = f"{custom_name}_{timestamp}.csv"
                else:
                    # Use default naming
                    base_name = current_data_info['filename'].replace('.csv', '') if current_data_info else 'predictions'
                    csv_filename = f"{base_name}_with_predictions_{timestamp}.csv"
                
                # Create comprehensive export DataFrame
                if uploaded_data is not None:
                    # Include all original features
                    export_df = uploaded_data.copy()
                    
                    # Add predictions
                    export_df['Predicted_Cost'] = predictions_df['Predicted']
                    
                    # Add metadata columns
                    if current_data_info:
                        export_df['model_used'] = current_data_info['model_used']
                        export_df['prediction_timestamp'] = current_data_info['timestamp']
                        export_df['source_file'] = current_data_info['filename']
                    
                    # Save to CSV
                    export_df.to_csv(csv_filename, index=False)
                    
                    print(f"✅ Complete dataset with predictions saved to: {csv_filename}")
                    print(f"📊 Exported {len(export_df)} rows with {len(export_df.columns)} columns")
                    
                    # Show column summary
                    feature_cols = [col for col in export_df.columns 
                                  if col not in ['Predicted_Cost', 'model_used', 'prediction_timestamp', 'source_file']]
                    print(f"📋 Includes:")
                    print(f"   • Original features: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
                    print(f"   • Predictions: Predicted_Cost")
                    print(f"   • Metadata: model_used, prediction_timestamp, source_file")
                    
                else:
                    # Fallback for packaged data testing
                    export_df = predictions_df.copy()
                    if current_data_info:
                        export_df['model_used'] = current_data_info['model_used']
                        export_df['prediction_timestamp'] = current_data_info['timestamp']
                    
                    export_df.to_csv(csv_filename, index=False)
                    print(f"✅ Predictions saved to: {csv_filename}")
                    print(f"📊 Exported {len(export_df)} predictions")
                    
            except Exception as e:
                print(f"❌ Error saving CSV: {e}")
                import traceback
                traceback.print_exc()
    else:
        with output:
            print("Please make predictions first")

predict_button.on_click(on_button_click)
summary_button.on_click(on_summary_click)
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
    widgets.HBox([summary_button, feature_selector]),  # Add this line
    save_plots_checkbox
])

export_section = widgets.VBox([
    widgets.HTML("<h3>💾 Export Options</h3>"),
    filename_input,
    widgets.HTML("<small>Leave blank for auto-generated filename</small>")
])

results_section = widgets.VBox([
    widgets.HTML("<h3>📊 Results</h3>"),
    output
])

# %%
# Optional: Instructions for users
instructions = widgets.HTML("""
<div style="background-color: #808080; padding: 15px; border-radius: 5px; margin: 10px 0;">
<h4>📋 Instructions:</h4>
<ol>
<li><strong>Upload your CSV file</strong> with building data (area, bldg_age, stories, etc.)</li>
<li><strong>Select a model</strong> (start with 'best_model')</li>
<li><strong>Make Predictions</strong> to generate retrofit cost estimates</li>
<li><strong>Plot Predictions</strong> to visualize results</li>
<li><strong>Optionally name your output file</strong> in the Export Options section</li>
<li><strong>Save to CSV</strong> to export complete dataset with predictions and metadata</li>
</ol>
<p><strong>Note:</strong> Ground truth values (ystruct19) are optional. The exported CSV includes all original features plus predictions and metadata.</p>
</div>
""")

display(instructions)

# %%
# Display all sections
display(upload_section)
display(action_section)
display(export_section)
display(results_section)
