import pytest
import pandas as pd
import numpy as np
from retrofit_cost_tool.data_utils import load_data, preprocess_data

class TestLoadData:
    def test_load_data_from_file(self, temp_csv_file):
        """Test loading data from CSV file"""
        data = load_data(temp_csv_file)
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert 'area' in data.columns

    def test_load_data_nonexistent_file(self):
        """Test error handling for nonexistent file"""
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.csv')

    def test_load_data_invalid_format(self):
        """Test error handling for invalid file format"""
        with pytest.raises(Exception):
            load_data('invalid_file.txt')

class TestPreprocessData:
    def test_preprocess_data_basic(self, sample_data_with_target):
        """Test basic data preprocessing"""
        features_string = ['p_obj_dummy', 'bldg_group_dummy', 'sp_dummy', 'historic_dummy']
        features_num = ['area', 'bldg_age', 'stories', 'seismicity_pga050', 'occup_cond']
        target = 'ystruct19'

        X, y = preprocess_data(sample_data_with_target, features_string, features_num, target)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 3
        assert len(X.columns) == len(features_string) + len(features_num)

    def test_preprocess_data_missing_columns(self, sample_data):
        """Test preprocessing with missing target column"""
        features_string = ['p_obj_dummy']
        features_num = ['area']
        target = 'nonexistent_target'

        with pytest.raises(KeyError):
            preprocess_data(sample_data, features_string, features_num, target)

    def test_preprocess_data_empty_dataframe(self):
        """Test preprocessing with empty DataFrame"""
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):
            preprocess_data(empty_data, [], [], 'target')
