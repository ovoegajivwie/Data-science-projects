#Student Grades Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/04/16/student-grades-prediction-with-machine-learning/

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/student-mat.csv')
dataset.head()
dataset.isnull().sum()

#selecting and splitting dataset
dataset = dataset[["G1", "G2", "G3", "studytime", "failures", "absences"]]
selected = "G3"
X = np.array(dataset.drop([selected], 1))
y = np.array(dataset[selected])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and fitting model
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(accuracy)

#making predictions

predictions = model.predict(X_test)
for i in range(len(predictions)):
    print(predictions[X], X_test[X], [y_test[X]])
