#Profit Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/04/29/profit-prediction-with-machine-learning/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/Startups.csv')
dataset.head()
dataset.isnull().sum()
dataset.describe()
correlation = dataset.corr()
print(correlation)

#visualizing the dataset
sns.heatmap(correlation, annot=True)
plt.show()

#selecting and splitting the dataset
X = dataset[["R&D Spend", "Administration", "Marketing Spend"]]
y = dataset["Profit"]
X = X.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and fitting model
model = LinearRegression()
model.fit(X_train, y_train)

#making predictions
ypred = model.predict(X_test)

#converting predictions to dataframe
data = pd.DataFrame({'predicted Profit': ypred.flatten()})
print(data.head())
