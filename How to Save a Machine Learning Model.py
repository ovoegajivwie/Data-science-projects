#How to Save a Machine Learning Model
#https://thecleverprogrammer.com/2021/05/13/how-to-save-a-machine-learning-model/

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle

#reading and viewing the dataset
dataset = pd.read_csv("student-mat.csv")
dataset.head()

#selecting and splitting the dataset
dataset = dataset[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
X = np.array(dataset.drop([predict], 1))
y = np.array(dataset[predict])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and fitting model
model = LinearRegression()
model.fit(X_train, y_train)

#saving the model
with open("pickle_model", "wb") as file:
    pickle.dump(model, file)
    
#viewing and using saved model
with open("pickle_model", "rb") as file:
    loaded_model = pickle.load(file)

predictions = loaded_model.predict(X_test)

for i in range(len(predictions)):
    print(predictions[X], X_test[X], [y_test[X]])
