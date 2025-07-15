# scripts/main.py
from retrofit_cost_tool.main import main

if __name__ == "__main__":
    best_model_name, best_model, model_metrics, X_valid, y_valid = main(verbose=True, save_models=True, save_metrics=True)
