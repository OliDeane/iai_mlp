from concat_stations import concat_files
#from data_prep import remove_nan
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pandas as pd
import json

# Prepare the data

full_data = concat_files()
feature_cols = list(full_data.drop('bikes', axis = 1).columns)
for i in full_data.columns:
    full_data = full_data[full_data[i].notna()]
X = full_data[feature_cols] # Features
y = full_data.bikes # Target Variable
# Turn weekday into 1-hot encoding
X = pd.get_dummies(X)
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)



# Create Decision Tree classifer object
criterion = "gini"
clf = DecisionTreeClassifier(criterion = criterion)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)


accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
model_type = "Decision Tree Classifier"
data = {"model_type": model_type, "criterion": criterion, "test_size" : test_size, "accuracy": accuracy}

with open('output_metrics.txt', 'a') as file:
     file.write(json.dumps(data)) # use `json.loads` to do the reverse

