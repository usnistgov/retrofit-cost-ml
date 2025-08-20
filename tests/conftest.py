import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

@pytest.fixture
def sample_data():
    """Create sample building data for testing"""
    return pd.DataFrame({
        'area': [5000, 8000, 3000],
        'bldg_age': [45, 30, 60],
        'stories': [3, 5, 2],
        'seismicity_pga050': [0.4, 0.6, 0.3],
        'p_obj_dummy': [1, 0, 1],
        'bldg_group_dummy': [1, 1, 0],
        'sp_dummy': [0, 1, 0],
        'occup_cond': [1, 2, 1],
        'historic_dummy': [0, 0, 1]
    })

@pytest.fixture
def sample_data_with_target(sample_data):
    """Sample data with target variable for training tests"""
    data = sample_data.copy()
    data['ystruct19'] = [125000, 180000, 95000]  # Sample costs
    return data

@pytest.fixture
def temp_csv_file(sample_data):
    """Create temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def invalid_data():
    """Create invalid data for testing error handling"""
    return pd.DataFrame({
        'area': [-1000, 8000],  # Negative area
        'bldg_age': [45, None],  # Missing value
        'stories': [3, 5],
        # Missing required columns
    })
