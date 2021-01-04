import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Import train_test_split function
from scipy import stats


def nan_to_median(full_data):

    station_ids = list(range(201,276)) # list of all station numbers


    bike_cols = ['bikes_3h_ago', 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'short_profile_3h_diff_bikes', 'short_profile_bikes']

    big_data = pd.DataFrame(columns = full_data.columns)

    for station in station_ids:
        df = full_data[full_data.station == station]
        for col in bike_cols:

            for i in range(1,max(df['weekhour'])):
                indices = df[df['weekhour'] == i].index.values.tolist()
                df.loc[indices[0], col] = df.loc[df['weekhour'] == i][col].median()

        big_data = big_data.append(df, ignore_index=True)
        
    for i in big_data.columns:
        big_data = big_data[big_data[i].notna()]

    big_data = big_data.apply(pd.to_numeric, errors='coerce')
    return big_data

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

def get_clean_data_rf(full_data):
    for i in full_data.columns:
        full_data = full_data[full_data[i].notna()]
    full_data = pd.get_dummies(full_data) # Turn weekday into 1-hot encoding
    feature_cols = list(full_data.drop('bikes', axis = 1).columns)

    y = np.array(full_data['bikes']) # array for target variable
    X = full_data[feature_cols] # Features
    X = np.array(X) # Turn into numpy array

def get_param_dist():

    """Gets optimum parametes for random forest regressor"""

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

def get_feature_df(full_data):
    """Returns a dataframe with just the most predictive features, bikes included"""

    df_features = ['longitude', 'timestamp', 'bikes_3h_ago', 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'airPressure.mb', 'weekhour', 'station', 'numDocks', 'hour', 'day', 'latitude', 'temperature.C', 'bikes']
    df = full_data[df_features]
    return df, df_features

def get_test_features(full_data):
    """Returns a dataframe with just the most predictive features, bikes excluded"""

    df_features = ['longitude', 'timestamp', 'bikes_3h_ago', 'full_profile_3h_diff_bikes', 'full_profile_bikes', 'short_profile_3h_diff_bikes', 'short_profile_bikes', 'airPressure.mb', 'weekhour', 'station', 'numDocks', 'hour', 'day', 'latitude', 'temperature.C']
    df = full_data[df_features]
    return df, df_features

def remove_outliers(df):
    """Remove outliers based on z_Scores above 3.00"""
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3.00).all(axis=1)
    new_full_data = df[filtered_entries]
    return new_full_data