#Microsoft Stock Price Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/06/21/microsoft-stock-price-prediction-with-machine-learning/

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#reading and viewing the dataset
#Visit Yahoo Finance
#Search for “MSFT”
#Click on “Historical Data”
#Click on “Download”
msft = pd.read_csv('MSFT.csv')
correlation = msft.corr()
print(msft.head())
print(correlation)


#plotting
plt.style.use('ggplot')
plt.plot(msft.Close)
plt.xlabel('Date')
plt.ylabel('Close')
plt.title("Microsoft Stock Prices")
plt.show()

sns.heatmap(correlation)
plt.show()

#selecting and splitting our data
X = msft[["Open", "High", "Low"]]
y = msft["Close"]

#converting and reshaping
X = X.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and fitting our model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#making predictions
predictions = model.predict(X_test)
dataset = pd.DataFrame({"Predicted Rate": predictions})
print(model.score(X_test, predictions))
print(dataset.head())
