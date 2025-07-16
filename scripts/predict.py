# scripts/predict.py
from retrofit_cost_tool.predict import main

if __name__ == "__main__":
    predictions = main()
    print(f"\nGenerated {len(predictions)} predictions")
