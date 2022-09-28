#Mobile Price Classification with Machine Learning
#https://thecleverprogrammer.com/2021/03/05/mobile-price-classification-with-machine-learning/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/mobile_prices.csv')
print(dataset.head())
dataset.info()
dataset.describe()
dataset.isnull().sum()
correlation = dataset.corr()
print(correlation)

#visualizing the dataset
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", linecolor='white', linewidths=1)
plt.show()

#Data Preparation
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X = StandardScaler().fit_transform(X)

#Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model creation
model = LogisticRegression()
model.fit(X_train, y_train)

#making predictions
predictions = model.predict(X_test)
print(predictions)

#checking model accuracy
accuracy = accuracy_score(y_test, predictions) * 100
print("Accuracy of the Logistic Regression Model: ",accuracy)

(unique, counts) = np.unique(predictions, return_counts=True)
price_range = np.asarray((unique, counts)).T
print(price_range)
