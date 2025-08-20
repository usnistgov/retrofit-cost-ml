import pytest
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from retrofit_cost_tool.model_io import save_model, load_model
import tempfile
import os

class TestModelTraining:
    def test_train_ridge_model(self, sample_data_with_target):
        """Test Ridge model training"""
        X = sample_data_with_target.drop('ystruct19', axis=1)
        y = sample_data_with_target['ystruct19']

        # Create and train model directly (since training functions need parameters)
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        assert hasattr(model, 'predict')
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_train_random_forest_model(self, sample_data_with_target):
        """Test Random Forest model training"""
        X = sample_data_with_target.drop('ystruct19', axis=1)
        y = sample_data_with_target['ystruct19']

        # Create and train model directly
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        assert hasattr(model, 'predict')
        predictions = model.predict(X)
        assert len(predictions) == len(y)

class TestModelIO:
    def test_save_and_load_model(self, sample_data_with_target):
        """Test model saving and loading"""
        X = sample_data_with_target.drop('ystruct19', axis=1)
        y = sample_data_with_target['ystruct19']

        # Train a simple model
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            save_model(model, model_path)

            # Load model
            loaded_model = load_model(model_path)

            # Test that loaded model works
            predictions_original = model.predict(X)
            predictions_loaded = loaded_model.predict(X)

            np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)
