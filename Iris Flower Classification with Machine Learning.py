#Iris Flower Classification with Machine Learning
#https://thecleverprogrammer.com/2021/10/17/iris-flower-classification-with-machine-learning/

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv')
dataset.head()
dataset.describe()
dataset.isnull().sum()
print("Target Labels", dataset["species"].unique())

#visualizing the dataset
sns.scatterplot(x='sepal_width', y='sepal_length', hue='species', data=dataset)
plt.show()

#selecting and splitting the dataset
X = dataset.drop("species", axis=1)
y = dataset["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and fitting model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#making predictions
predict = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(predict)
print("Prediction: {}".format(prediction))
