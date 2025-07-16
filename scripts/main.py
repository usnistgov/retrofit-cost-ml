# scripts/main.py
from retrofit_cost_tool.main import main

if __name__ == "__main__":
    # Use defaults optimized for testing and development
    best_model_name, best_model, model_metrics, X_valid, y_valid = main(
        verbose=True,
        save_models=False,      # Set to False for testing
        save_metrics=False,     # Set to False for testing
        suppress_warnings=True  # Add this parameter
    )
    print(f"\nTraining completed. Best model: {best_model_name}")
