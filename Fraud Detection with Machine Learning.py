#Fraud Detection with Machine Learning
#https://thecleverprogrammer.com/2020/08/04/fraud-detection-with-machine-learning/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score

#reading and viewing files
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/payment_fraud.csv')
print(dataset.head())
dataset.isnull().sum()
dataset.info()
dataset.describe()

#finding the correlation
correlation = dataset.corr()
print(correlation)

#selecting and splitting dataset
dataset = dataset.drop('paymentMethod', axis=1)
X = dataset.drop('label', axis=1)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#model creation and fitting
model = LogisticRegression()
model.fit(X_train, y_train)

#making predictions
predictions = model.predict(X_test)
print(accuracy_score(predictions, y_test))

#compare test set predictions with ground truth labels
print(confusion_matrix(y_test, predictions))
