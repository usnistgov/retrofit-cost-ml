import pytest
import numpy as np
import pandas as pd
from retrofit_cost_tool import predict

class TestPredict:
    def test_predict_basic(self, sample_data):
        """Test basic prediction functionality"""
        predictions = predict(sample_data, model_name='best_model')

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        assert all(pred > 0 for pred in predictions)  # Costs should be positive

    def test_predict_different_models(self, sample_data):
        """Test prediction with different models"""
        models = ['ridge_model', 'random_forest_model', 'best_model']

        for model_name in models:
            predictions = predict(sample_data, model_name=model_name)
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == len(sample_data)

    def test_predict_invalid_model(self, sample_data):
        """Test error handling for invalid model name"""
        # Based on the error, it tries to load a file that doesn't exist
        with pytest.raises(FileNotFoundError):
            predict(sample_data, model_name='nonexistent_model')

    def test_predict_missing_columns(self):
        """Test error handling for missing required columns"""
        incomplete_data = pd.DataFrame({'area': [5000]})  # Missing required columns

        # This will likely raise a KeyError or similar when preprocessing
        with pytest.raises(Exception):  # Use generic Exception since exact error type may vary
            predict(incomplete_data)

    def test_predict_empty_data(self):
        """Test error handling for empty data"""
        empty_data = pd.DataFrame()

        with pytest.raises(Exception):
            predict(empty_data)

    def test_predict_single_row(self, sample_data):
        """Test prediction with single building"""
        single_building = sample_data.iloc[[0]].copy()  # Add .copy() to avoid warning
        predictions = predict(single_building)

        assert len(predictions) == 1
        assert predictions[0] > 0
