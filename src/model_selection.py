# model_selection.py
"""
Module for performing model selection using cross-validation.

This module provides a function for selecting the best-performing model from a dictionary of models.

Functions:
    model_selection(models, X, y, cv=5)
        Perform model selection using cross-validation.

        Args:
        - models (dict): Dictionary of models to evaluate.
        - X (array-like): Feature data.
        - y (array-like): Target data.
        - cv (int, optional): Number of folds for cross-validation. Defaults to 5.

        Returns:
        - best_model_name (str): Name of the best-performing model.
        - best_model (object): Best-performing model.
"""

from sklearn.model_selection import cross_val_score
import numpy as np

def model_selection(models, X, y, cv=5):
    """
    Perform model selection using cross-validation.

    Args:
    - models (dict): Dictionary of models to evaluate.
    - X (array-like): Feature data.
    - y (array-like): Target data.
    - cv (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
    - best_model_name (str): Name of the best-performing model.
    - best_model (object): Best-performing model.
    """
    scores = {}
    for model_name, model in models.items():
        scores[model_name] = np.mean(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
    
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    
    return best_model_name, best_model
