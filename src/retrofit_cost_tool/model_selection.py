# src/retrofit_cost_tool/model_selection.py

from sklearn.model_selection import KFold, train_test_split
import numpy as np
import inspect
from .model_utils import evaluate_model

def model_selection(model_train_funcs, hyperparams, X, y, outer_cv=5, inner_cv=5, random_state=42, verbose=False):
    """
    Perform model selection using nested cross-validation.

    Args:
    - model_train_funcs (dict): Dictionary of model training functions and hyperparameter grids.
    - hyperparams (None): Not used.
    - X (array-like): Feature data.
    - y (array-like): Target data.
    - outer_cv (int, optional): Number of folds for outer cross-validation. Defaults to 5.
    - inner_cv (int, optional): Number of folds for inner cross-validation. Defaults to 5.
    - random_state (int, optional): Random state for cross-validation. Defaults to 42.
    - verbose (bool, optional): Whether to print evaluation metrics during model selection. Defaults to False.

    Returns:
    - best_model_name (str): Name of the best-performing model.
    - best_model (object): Best-performing model.
    - model_metrics (dict): Dictionary of evaluation metrics for each model.
    """
    scores = {}
    model_metrics = {}
    metrics = ['rmse', 'mae', 'mape']
    for model_name, (model_train_func, hyperparam_grid) in model_train_funcs.items():
        if verbose:
            print(f'Evaluating {model_name}...')
        metric_values = {metric: [] for metric in metrics}
        for train_idx, val_idx in KFold(n_splits=outer_cv, shuffle=True, random_state=random_state).split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Perform hyperparameter tuning using inner CV
            if 'cv' in inspect.signature(model_train_func).parameters:
                model = model_train_func(X_train, y_train, *hyperparam_grid, cv=inner_cv)
            else:
                model = model_train_func(X_train, y_train, *hyperparam_grid)

            # Evaluate the model on the validation set
            rmse, mae, mape = evaluate_model(model, X_val, y_val)
            metric_values['rmse'].append(rmse)
            metric_values['mae'].append(mae)
            metric_values['mape'].append(mape)

        scores[model_name] = np.mean(metric_values['rmse'])
        model_metrics[model_name] = {metric: {'mean': np.mean(metric_values[metric]), 'std': np.std(metric_values[metric])} for metric in metrics}
        if verbose:
            print(f'{model_name.capitalize()} RMSE: {scores[model_name]:.4f}')

    best_model_name = min(scores, key=scores.get)
    best_model_train_func, best_hyperparam_grid = model_train_funcs[best_model_name]

    # Refit the best model on the entire dataset
    best_model = best_model_train_func(X, y, *best_hyperparam_grid)

    return best_model_name, best_model, model_metrics
