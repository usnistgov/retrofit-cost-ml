# model_selection.py
"""
Module for performing hyperparameter tuning and model selection using nested cross-validation.

This module provides a function for selecting the best-performing model from a dictionary of models.
"""
from .model_utils import evaluate_model
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import numpy as np

def model_selection(model_train_funcs, hyperparams, X, y, outer_cv=5, inner_cv=5, verbose=False):
    """
    Perform model selection using nested cross-validation.

    Args:
    - model_train_funcs (dict): Dictionary of model training functions and hyperparameter grids.
    - hyperparams (None): Not used.
    - X (array-like): Feature data.
    - y (array-like): Target data.
    - outer_cv (int, optional): Number of folds for outer cross-validation. Defaults to 5.
    - inner_cv (int, optional): Number of folds for inner cross-validation. Defaults to 5.

    Returns:
    - best_model_name (str): Name of the best-performing model.
    - best_model (object): Best-performing model.
    """
    scores = {}
    model_metrics = {}
    for model_name, (model_train_func, hyperparam_grid) in model_train_funcs.items():
        if verbose:
            print(f'Evaluating {model_name}...')
        scores_model = []
        for train_idx, val_idx in KFold(n_splits=outer_cv, shuffle=True, random_state=42).split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Perform hyperparameter tuning using inner CV
            model = model_train_func(X_train, y_train, *hyperparam_grid, cv=inner_cv)

            # Evaluate the model on the validation set
            rmse, mae, mape = evaluate_model(model, X_val, y_val)
            scores_model.append(rmse)

        scores[model_name] = np.mean(scores_model)
        model_metrics[model_name] = {'rmse': np.mean(scores_model), 'rmse_std': np.std(scores_model)}
        if verbose:
            print(f'{model_name.capitalize()} RMSE: {scores[model_name]:.4f}')

    best_model_name = min(scores, key=scores.get)
    best_model_train_func, best_hyperparam_grid = model_train_funcs[best_model_name]

    # Refit the best model on the entire dataset
    best_model = best_model_train_func(X, y, *best_hyperparam_grid)

    return best_model_name, best_model, model_metrics
