# scripts/main.py
from retrofit_cost_tool.main import main

if __name__ == "__main__":
    # Use consistent defaults with the main module
    best_model_name, best_model, model_metrics, X_valid, y_valid = main(
        verbose=True, 
        save_models=True,  # Changed to True for consistency
        save_metrics=True   # Changed to True for consistency
    )
    print(f"\nTraining completed. Best model: {best_model_name}")
