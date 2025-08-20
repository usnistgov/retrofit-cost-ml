# Retrofit Cost Prediction Tool

A Python package for predicting seismic retrofit costs using pre-trained machine
learning models.

## Overview

This tool provides pre-trained machine learning models to predict the cost of
seismic retrofits for buildings. Users can select any of the models to make
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

### 3. Interactive Jupyter Notebook

For users who prefer a graphical interface, use our interactive notebook:

```bash
# Launch Jupyter and open the prediction notebook
jupyter notebook notebooks/retrofit-cost-tool-predict.ipynb
```

**Features:**
- 📁 **File upload widget** - drag and drop your CSV files
- 🤖 **Model selection dropdown** - choose from all available models
- 📊 **Interactive plotting** - visualize predictions vs actual values
- 💾 **CSV export** - save predictions with original features and metadata
- 📈 **Feature analysis** - summary statistics by building characteristics

## Data Format

Your input data should be a CSV file with the following columns:

### Required Columns

| Column | Description | Type | Example | Valid Range |
|--------|-------------|------|---------|-------------|
| `area` | Building area (sq ft) | Numeric | 5000 | > 0 |
| `bldg_age` | Building age (years) | Numeric | 45 | ≥ 0 |
| `stories` | Number of stories | Numeric | 3 | ≥ 1 |
| `seismicity_pga050` | Peak ground acceleration | Numeric | 0.4 | 0-1 |
| `p_obj_dummy` | Performance objective | Binary | 1 | 0 or 1 |
| `bldg_group_dummy` | Building type | Binary | 1 | 0 or 1 |
| `sp_dummy` | Structural performance | Binary | 0 | 0 or 1 |
| `occup_cond` | Occupancy condition | Numeric | 1 | 1-3 |
| `historic_dummy` | Historic designation | Binary | 0 | 0 or 1 |

### Optional Columns
- `ystruct19`: Actual retrofit costs (for model validation and comparison)
- `building_id`: Building identifier (for tracking and reporting)

### Sample Data
Download example data: [synthetic_data.csv](src/retrofit_cost_tool/data/synthetic_data.csv)


## Model Information

The package includes several pre-trained models:

- **Ridge Regression**: Linear model with L2 regularization
- **Elastic Net**: Linear model with L1 and L2 regularization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Boosted ensemble method
- **OLS**: Ordinary Least Squares regression
- **GLM Gamma**: Generalized Linear Model with Gamma distribution

The `best_model` automatically selects the best performing model based on cross-validation.

## Training Data

### Dataset Overview

The models were trained on a comprehensive dataset of seismic
retrofit projects to ensure robust predictions across different building types
and seismic conditions.[^1]

**Training Dataset Characteristics:**

- **Size**: 1526 buildings
- **Geographic Coverage**: United States
- **Building Types**: Office buildings, residential structures, mixed-use, historic buildings
- **Seismic Zones**: Range of seismic hazard levels
- ** Performance Objectives**: Mainly Life Safety (LS) but also some Damage
  Control (DC) and Immediate Occupancy (IO)
- **Target**: Structural retrofit cost (per square foot)

### Data Preprocessing

- Feature scaling and normalization
- Categorical variable encoding
- Interaction term creation

### Limitations

- Training data is comprehensive but outdated and does not represent current
  technical practices for seismic retrofits
- Performance may vary in areas with different building codes
- Costs scaled to 2019 USD. Users should adjust for inflation and local market
  factors

### Synthetic Data for Testing

For users who want to test the package without real building data:

```python
from retrofit_cost_tool.data_utils import load_data
from importlib import resources

# Load synthetic test data
with resources.path('retrofit_cost_tool.data', 'synthetic_data.csv') as data_path:
    test_data = load_data(str(data_path))
```

**Synthetic Data Features:**

- Realistic building parameter distributions, based on original (training) data
- Covers full range of model inputs
- Includes ground truth target for validation
- Safe for public sharing and testing


[^1]: FEMA (1994). Typical Costs for Seismic Rehabilitation of Existing Buildings. Vol 1: Summary. FEMA 156, Second Edition.


## Examples

### Basic Building Analysis

```python
import pandas as pd
from retrofit_cost_tool import predict

# Create sample building data
building_data = pd.DataFrame({
    'area': [5000, 8000, 12000],
    'bldg_age': [45, 30, 60], 
    'stories': [3, 5, 8],
    'seismicity_pga050': [0.4, 0.6, 0.5],
    'p_obj_dummy': [1, 0, 1],
    'bldg_group_dummy': [1, 1, 1],
    'sp_dummy': [0, 1, 0],
    'occup_cond': [1, 2, 2],
    'historic_dummy': [0, 0, 1]
})

# Get predictions
costs = predict(building_data, model_name='best_model')
print(f"Predicted costs: ${costs[0]:,.0f}, ${costs[1]:,.0f}, ${costs[2]:,.0f}")
```

### Portfolio Analysis

```python
# Load building portfolio
portfolio = load_data('my_buildings.csv')

# Compare different models
for model in ['ridge_model', 'random_forest_model', 'best_model']:
    predictions = predict(portfolio, model_name=model)
    total_cost = sum(predictions)
    print(f"{model}: Total portfolio cost = ${total_cost:,.0f}")
```


## Development

### Project Structure

```
retrofit-cost-tool/
├── src/retrofit_cost_tool/
│   ├── __init__.py
│   ├── main.py                          # Model training
│   ├── predict.py                       # Prediction functionality
│   ├── data_utils.py                    # Data loading and preprocessing
│   ├── model_utils.py                   # Model training functions
│   ├── model_io.py                      # Model saving/loading
│   ├── model_selection.py               # Model selection logic
│   ├── plot_utils.py                    # Visualization utilities
│   ├── data/                            # Training and synthetic data
│   └── models/                          # Pre-trained models
├── scripts/
│   ├── main.py                          # Training script
│   └── predict.py                       # Prediction script
├── notebooks/
│   ├── retrofit-cost-tool-predict.ipynb # Interactive prediction interface
├── pyproject.toml                       # Package configuration
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

## Troubleshooting

### Common Issues

**"Column not found" errors:**
- Ensure your CSV has all required columns with exact names (case-sensitive)
- Use the interactive notebook for automatic data validation

**"Model not found" errors:**
- Available models: `ridge_model`, `elastic_net_model`, `random_forest_model`, `gradient_boosting_model`, `ols_model`, `glm_gamma_model`, `best_model`

**Unexpected predictions:**
- Verify data ranges (area > 0, age ≥ 0, dummy variables are 0/1)
- Check for missing values in your data

**Jupyter notebook widget issues:**
- Install widget dependencies: `pip install ipywidgets`
- Enable widgets: `jupyter nbextension enable --py widgetsnbextension`
- Restart Jupyter after installation

### Getting Help

- Check data format with the built-in validation
- Contact: juan.fung@nist.gov

## License

NIST-developed software is provided by NIST as a public service. You may use,
copy, and distribute copies of the software in any medium, provided that you
keep intact this entire notice. You may improve, modify, and create derivative
works of the software or any portion of the software, and you may copy and
distribute such modifications or works. Modified works should carry a notice
stating that you changed the software and should note the date and nature of any
such change. Please explicitly acknowledge the National Institute of Standards
and Technology as the source of the software.

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF
ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY OPERATION OF LAW, INCLUDING,
WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE, NON-INFRINGEMENT, AND DATA ACCURACY. NIST NEITHER REPRESENTS
NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR
ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE
ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF,
INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR
USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and
distributing the software and you assume all risks associated with its use,
including but not limited to the risks and costs of program errors, compliance
with applicable laws, damage to or loss of data, programs or equipment, and the
unavailability or interruption of operation. This software is not intended to be
used in any situation where a failure could cause risk of injury or damage to
property. The software developed by NIST employees is not subject to copyright
protection within the United States.

## Author

Juan F. Fung (juan.fung@nist.gov)

## Version

1.0.0
