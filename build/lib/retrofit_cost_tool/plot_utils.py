# src/retrofit_cost_tool/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(predictions_df, actual_values, plot_scatter=True, plot_histograms=True, save_plots=False, save_dir='.'):
    if plot_histograms:
        sns.histplot(predictions_df['Predicted'], kde=True, label='Predicted Costs')
        sns.histplot(actual_values, kde=True, label='Actual Costs', color='red')
        plt.legend()
        plt.title('Cost Predictions')
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        if save_plots:
            plt.savefig(os.path.join(save_dir, 'cost_predictions.png'), bbox_inches='tight', dpi=300)
        plt.show()

    if plot_scatter:
        plt.scatter(actual_values, predictions_df['Predicted'])
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        if save_plots:
            plt.savefig(os.path.join(save_dir, 'predicted_vs_actual.png'), bbox_inches='tight', dpi=300)
        plt.show()
