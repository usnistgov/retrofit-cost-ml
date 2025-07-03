# Seismic Retrofit Cost Prediction Tool

This repository contains a Python script that allows users to load their own
data and make predictions using pre-trained models.

## Installation

To use this script, you'll need to have Python installed on your system. You'll
also need to install the required dependencies, which can be done using pip:

```bash
pip install -r requirements.txt
```

## Usage

To use the script, simply run it from the command line:

```bash
python predict.py
```

The script will prompt you to enter the path to your data file and the name of
the model to use. Follow the prompts to make predictions.

## Input Data

The input data should be a CSV file containing the following columns:

* `seismicity_pga050`
* `p_obj_dummy`
* `bldg_group_dummy`
* `sp_dummy`
* `occup_cond`
* `historic_dummy`
* `area`
* `bldg_age`
* `stories`

## Outputs

The script will output the predicted seismic retrofit costs for the input data.

## Models

The script comes with several pre-trained models, including:

* Ridge regression
* Elastic Net
* Random Forest
* Gradient Boosting

You can select the model to use by entering the corresponding name when prompted.
