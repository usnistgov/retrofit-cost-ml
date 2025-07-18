# src/retrofit_cost_tool/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_predictions(predictions_df, actual_values=None, plot_scatter=True, plot_histograms=True, save_plots=False, save_dir='.'):
    if plot_histograms:
        sns.histplot(predictions_df['Predicted'], kde=True, label='Predicted Costs')
        
        # Only plot actual values histogram if actual_values is provided and different from predictions
        if actual_values is not None and not predictions_df['Predicted'].equals(actual_values):
            sns.histplot(actual_values, kde=True, label='Actual Costs', color='red')
            plt.legend()
            plt.title('Predicted vs Actual Cost Distribution')
        else:
            plt.title('Predicted Cost Distribution')
            
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        if save_plots:
            plt.savefig(os.path.join(save_dir, 'cost_predictions.png'), bbox_inches='tight', dpi=300)
        plt.show()

    if plot_scatter and actual_values is not None and not predictions_df['Predicted'].equals(actual_values):
        plt.scatter(actual_values, predictions_df['Predicted'])
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        
        # Add perfect prediction line
        min_val = min(min(actual_values), min(predictions_df['Predicted']))
        max_val = max(max(actual_values), max(predictions_df['Predicted']))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        plt.legend()
        
        if save_plots:
            plt.savefig(os.path.join(save_dir, 'predicted_vs_actual.png'), bbox_inches='tight', dpi=300)
        plt.show()
