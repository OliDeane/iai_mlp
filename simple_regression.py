import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load in dataset
path = os.getcwd()
data = pd.read_csv(os.path.join(path, "morebikes2020/Train/Train/station_201_deploy.csv"))

data_dict = {"bikes_3h_ago": data["bikes_3h_ago"], "bikes": data["bikes"]}

df = pd.DataFrame(data = data_dict)
reg_data = df[pd.notnull(df["bikes_3h_ago"])]
reg_data = reg_data[pd.notnull(reg_data["bikes"])]

# Train-test split
train, test = train_test_split(reg_data, random_state=42)

# Generate Linear Regression Model
model = LinearRegression(fit_intercept=True)
model.fit(train[["bikes_3h_ago"]], train["bikes"])

print("Accuracy: {}".format(model.score(test[["bikes_3h_ago"]], test["bikes"])))
print("Training set: {}".format(len(train)))
print("Test set: {}".format(len(test)))

"""
#Make predictions with the model
x_fit = pd.DataFrame([data["bikes_3h_ago"].min(), data["bikes_3h_ago"].max()])
y_pred = model.predict(x_fit)

fig, ax = plt.subplots()
data.plot.scatter("bikes_3h_ago", "bikes", ax=ax)
ax.plot(x_fit[0], y_pred, linestyle=":")
#plt.show()

print(" Model gradient: ", model.coef_[0])
print("Model intercept:", model.intercept_)
"""