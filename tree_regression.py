from concat_stations import concat_files
import pandas as pd
import json
import numpy as np
from data_prep import get_preprocessed_data
import matplotlib.pyplot as plt

#from data_prep import remove_nan
from sklearn.tree import DecisionTreeRegressor # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# Prepare the data

full_data = concat_files()
feature_cols = list(full_data.drop('bikes', axis = 1).columns)
X_train, X_test, y_train, y_test, test_size = get_preprocessed_data(full_data, feature_cols)


criterion = "mse" # Create Decision Tree classifer object
regressor = DecisionTreeRegressor(random_state=0, criterion = criterion)
regressor = regressor.fit(X_train,y_train) # Train Decision Tree Classifer
y_pred = regressor.predict(X_test) #Predict the response for test dataset


# evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# Save model performance
model_type = "Decision Tree Regressor"


data = {"model_type": model_type, "criterion": criterion, "test_size" : test_size, "mean_abs_err": metrics.mean_absolute_error(y_test, y_pred),
"mean_sqrd_err" : metrics.mean_squared_error(y_test, y_pred), "Root_MSE" : np.sqrt(metrics.mean_squared_error(y_test, y_pred))}
with open('output_metrics.txt', 'a') as file:
     file.write(json.dumps(data)) # use `json.loads` to do the reverse

