import pandas as pd
import numpy as np
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


def get_param_dist():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    return random_grid