import pandas as pd
import numpy as np
from concat_stations import concat_files
import data_prep

import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# Prepare the data
full_data = concat_files() # Join all station data
X, y, feature_cols = data_prep.get_clean_data_rf(full_data) # Remove string data from weekday col and seaprate into X and y data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)



"""
Now train the model
"""

# Instantiate model with optimised hyperparamters decision trees
rf = RandomForestRegressor(n_estimators = 400, min_samples_split = 2, min_samples_leaf = 1, 
max_features = 'sqrt', max_depth = None, bootstrap = False)
# Train the model on training data
rf.fit(X_train, y_train)



"""
Evaluation
"""
y_pred = rf.predict(X_test) # Use the forest's predict method on the test data
print("mae {}".format(metrics.mean_absolute_error(y_test, y_pred)))
print("r2_error {}".format(metrics.r2_score(y_test, y_pred)))

# Save model performance
model_type = "Random Forest (tuned params)"
data = {"model_type": model_type, "params": rf.get_params(), "mean_abs_err": metrics.mean_absolute_error(y_test, y_pred),
"mean_sqrd_err" : metrics.mean_squared_error(y_test, y_pred), "Root_MSE" : np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
"r2_error" : metrics.r2_score(y_test, y_pred)}

# with open('output_metrics.txt', 'a') as file:
#      file.write(json.dumps(data)) # use `json.loads` to do the reverse