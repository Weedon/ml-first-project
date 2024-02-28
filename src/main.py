import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn.utils import shuffle

## Linear Regression ##

# Import data as dataframes, cut down to only selected columns
student_data = pd.read_csv("data/student-mat.csv", sep=";")
student_data = student_data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Predict is called "label", what we want the model to determine
predict = "G3"
x = np.array(student_data.drop([predict], axis=1))
y = np.array(student_data[predict])

# Separate training and testing data, 90% train and 10% test
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size = 0.1)

# Use fit our data into a linear model (y = mx + b) and record the accuracy
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print(accuracy)

