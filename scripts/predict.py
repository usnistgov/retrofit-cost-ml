# scripts/predict.py
from retrofit_cost_tool.predict import main

if __name__ == "__main__":
    predictions = main()
    if predictions is not None:
        print(f"\nGenerated {len(predictions)} predictions")
    else:
        print("Prediction failed.")
