# Retrofit Cost Tool

A Python package for predicting seismic retrofit costs using machine learning models.

## Overview

This tool provides machine learning models to predict the cost of seismic
retrofits for buildings. It includes pre-trained models and allows users to make
predictions on their own data.

## Installation

### From GitLab (Recommended)

```bash
pip install git+https://sgitlab.nist.gov/jff/retrofit-cost-tool.git
```

## Requirements

- Python 3.12
- numpy==1.26.4
- pandas==2.3.0
- scikit-learn==1.7.0
- joblib==1.5.1
- matplotlib==3.10.3
- seaborn==0.13.2

## Usage

### 1. Command-Line Scripts

#### Making Predictions

```bash
# Use default synthetic data
python scripts/predict.py

# Use your own data file
python scripts/predict.py path/to/your/data.csv

# Use specific model
python scripts/predict.py path/to/your/data.csv random_forest_model
```

Available models: `ridge_model`, `elastic_net_model`, `random_forest_model`, `gradient_boosting_model`, `ols_model`, `glm_gamma_model`, `best_model`

#### Training Models

```bash
# Train models (saves to package models directory)
python scripts/main.py
```

### 2. Programmatic API

```python
from retrofit_cost_tool import load_data, predict, main
import pandas as pd

# Load your data
data = load_data('path/to/your/data.csv')

# Make predictions
predictions = predict(data, model_name='best_model')

# Train new models
best_model_name, best_model, metrics, X_valid, y_valid = main(
    verbose=True,
    save_models=True,
    save_metrics=True
)
```

## Data Format

Your input data should be a CSV file with the following columns:

### Required Features:
- `seismicity_pga050`: Seismic hazard measure
- `p_obj_dummy`: Building performance objective (dummy variable)
- `bldg_group_dummy`: Building group classification (dummy variable)
- `sp_dummy`: Structural performance dummy variable
- `occup_cond`: Occupancy condition
- `historic_dummy`: Historic building designation (dummy variable)
- `area`: Building area (square feet)
- `bldg_age`: Building age (years)
- `stories`: Number of stories

### Target Variable (for training):
- `ystruct19`: Structural retrofit cost (target variable)

## Model Information

The package includes several pre-trained models:

- **Ridge Regression**: Linear model with L2 regularization
- **Elastic Net**: Linear model with L1 and L2 regularization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Boosted ensemble method
- **OLS**: Ordinary Least Squares regression
- **GLM Gamma**: Generalized Linear Model with Gamma distribution

The `best_model` automatically selects the best performing model based on cross-validation.

## Development

### Project Structure

```
retrofit-cost-tool/
├── src/retrofit_cost_tool/
│   ├── __init__.py
│   ├── main.py              # Model training
│   ├── predict.py           # Prediction functionality
│   ├── data_utils.py        # Data loading and preprocessing
│   ├── model_utils.py       # Model training functions
│   ├── model_io.py          # Model saving/loading
│   ├── model_selection.py   # Model selection logic
│   ├── plot_utils.py        # Visualization utilities
│   ├── data/               # Training and synthetic data
│   └── models/             # Pre-trained models
├── scripts/
│   ├── main.py             # Training script
│   └── predict.py          # Prediction script
├── notebooks/              # Jupyter notebooks
├── pyproject.toml          # Package configuration
└── README.md
```

### Building from Source

```bash
# Install build dependencies
pip install build

# Build the package
python -m build

# Install in development mode
pip install -e .
```

## License

MIT License

## Author

Juan F. Fung (juan.fung@nist.gov)

## Version

1.0.0
