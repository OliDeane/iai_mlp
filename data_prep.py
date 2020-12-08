import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function


def remove_nan(full_data):
    '''Returned data without NaNs.'''
    for i in full_data.columns:
        full_data = full_data[full_data[i].notna()]
    return full_data

def get_preprocessed_data(full_data, feature_cols):
    for i in full_data.columns:
        full_data = full_data[full_data[i].notna()]
    X = full_data[feature_cols] # Features
    y = full_data.bikes # Target Variable
    # Turn weekday into 1-hot encoding
    X = pd.get_dummies(X)
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    return X_train, X_test, y_train, y_test, test_size