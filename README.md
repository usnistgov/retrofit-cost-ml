# Seismic Retrofit Cost Tool

A Python package for predicting seismic retrofit costs.

## Installation

To install the package, run the following command:

```bash
pip install .
```

## Usage

To use the package, you can import the necessary modules and functions. For example:

```python
from retrofit_cost_tool import load_data, preprocess_data

data = load_data('data/srce_train.csv')
X, y = preprocess_data(data, features_string, features_num, target)
```

## Scripts

The package includes several scripts that demonstrate how to use the package. These scripts are located in the `scripts/` directory.

* `main.py`: Trains and saves machine learning models using the training data.
* `predict.py`: Allows users to load their own data and make predictions using pre-trained models.

## Notebooks

The package includes several Jupyter Notebooks that demonstrate how to use the package. These notebooks are located in the `notebooks/` directory.

## Data

The package includes example data in the `data/` directory.

## Models

The package includes pre-trained models in the `models/` directory.

## Requirements

The package requires the following dependencies:

* `pandas`
* `numpy`
* `scikit-learn`
* `joblib`

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```
