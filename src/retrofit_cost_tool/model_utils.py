# model_utils.py (updated)

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, GammaRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def train_ols_model(X_train, y_train):
    """Train an OLS model."""
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    return ols

def train_glm_gamma_model(X_train, y_train):
    """Train Gamma GLM model."""
    glm_gamma = GammaRegressor()
    glm_gamma.fit(X_train, y_train)
    return glm_gamma

def train_ridge_model(X_train, y_train, alpha_grid, cv=10):
    """Train a Ridge regression model with hyperparameter tuning."""
    param_grid = {'alpha': alpha_grid}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_elastic_net_model(X_train, y_train, alpha_grid, l1_ratio_grid, cv=10):
    """Train an Elastic Net model with hyperparameter tuning."""
    param_grid = {'alpha': alpha_grid, 'l1_ratio': l1_ratio_grid}
    elastic_net = ElasticNet(max_iter=10000, tol=1e-3)
    grid_search = GridSearchCV(elastic_net, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_random_forest_model(X_train, y_train, n_estimators_grid, max_depth_grid, cv=10):
    """Train a Random Forest model with hyperparameter tuning."""
    param_grid = {'n_estimators': n_estimators_grid, 'max_depth': max_depth_grid}
    random_forest = RandomForestRegressor()
    grid_search = GridSearchCV(random_forest, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_gradient_boosting_model(X_train, y_train, n_estimators_grid, max_depth_grid, cv=10):
    """Train a Gradient Boosting model with hyperparameter tuning."""
    param_grid = {'n_estimators': n_estimators_grid, 'max_depth': max_depth_grid}
    gradient_boosting = GradientBoostingRegressor()
    grid_search = GridSearchCV(gradient_boosting, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_valid, y_valid):
    """Evaluate the performance of a model."""
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae = mean_absolute_error(y_valid, y_pred)
    mape = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100
    return rmse, mae, mape
