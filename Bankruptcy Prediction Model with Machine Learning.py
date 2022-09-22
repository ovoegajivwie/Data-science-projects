#Bankruptcy Prediction Model with Machine Learning
#https://thecleverprogrammer.com/2021/03/13/bankruptcy-prediction-model-with-machine-learning/

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/bank.csv')
dataset.head()
dataset.isnull().sum()
correlation = dataset.corr()
print(correlation)

#visualizing the dataset
plt.figure(figsize = (32,10))
sns.heatmap(correlation)
plt.show()

#selecting and splitting the dataset
X = dataset.drop(["Bankrupt?"], axis="columns")
y = dataset["Bankrupt?"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model creation and fitting
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
