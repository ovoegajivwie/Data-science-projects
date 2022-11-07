#Click-Through Rate Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/01/24/click-through-rate-prediction-with-machine-learning/
#https://www.kaggle.com/fayomi/advertising/download

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#reading and viewing dataset
data = pd.read_csv('advertising.csv')
print(data.head())
print(data.isnull().sum())
print(data.describe())
print(data.info())
correlation = data.corr()
print(correlation)
print(data.columns)

#selecting cols
x = data.iloc[:,0:7]
X = x.drop(['Ad Topic Line','City'], axis=1)
y = data.iloc[:,9]

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#model creation and fitting
Lr_model =LogisticRegression(C=0.01,random_state=0)
Lr_model.fit(X_train, y_train)

#making predictions
y_pred = Lr_model.predict(X_test)
print(y_pred)
y_pred_proba = Lr_model.predict_proba(X_test)
print(y_pred_proba)

#getting accuracy
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
