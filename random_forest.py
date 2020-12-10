import pandas as pd
import numpy as np
from concat_stations import concat_files

import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# Prepare the data

full_data = concat_files()
for i in full_data.columns:
    full_data = full_data[full_data[i].notna()]
full_data = pd.get_dummies(full_data) # Turn weekday into 1-hot encoding
feature_cols = list(full_data.drop('bikes', axis = 1).columns)

y = np.array(full_data['bikes']) # array for target variable
X = full_data[feature_cols] # Features
X = np.array(X) # Turn into numpy array

print(X.shape)
print(y.shape)

test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)


"""
Now establish a baseline
Baseline measure that we want to be with our model. Here that could be bikes_3h_ago?
"""

# The baseline predictions are the historical averages
baseline_preds = X_train[:, feature_cols.index('full_profile_bikes')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - y_train)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))


"""
Now train the model
"""

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)

"""
Evaluation
"""
y_pred = rf.predict(X_test) # Use the forest's predict method on the test data
errors = abs(y_pred - y_test) # Calculate the absolute errors

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Save model performance
model_type = "Random Forest"
criterion = "NA"
data = {"model_type": model_type, "criterion": criterion, "test_size" : test_size, "mean_abs_err": metrics.mean_absolute_error(y_test, y_pred),
"mean_sqrd_err" : metrics.mean_squared_error(y_test, y_pred), "Root_MSE" : np.sqrt(metrics.mean_squared_error(y_test, y_pred))}
with open('output_metrics.txt', 'a') as file:
     file.write(json.dumps(data)) # use `json.loads` to do the reverse