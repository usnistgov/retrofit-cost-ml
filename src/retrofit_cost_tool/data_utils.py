# data_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data, features_string, features_num, target):
    """Preprocess data by encoding categorical variables and scaling numerical variables."""
    # Encode categorical variables
    data[features_string] = data[features_string].apply(LabelEncoder().fit_transform)
    
    # Split data into features and target
    X = data[features_string + features_num]
    y = data[target]
    
    # Scale numerical variables
    scaler = StandardScaler()
    X[features_num] = scaler.fit_transform(X[features_num])
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=1234):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
