#Car Price Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/08/04/car-price-prediction-with-machine-learning/

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#reading and viewing the data
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv')
dataset.head()
dataset.isnull().sum()
print(dataset.info())
print(dataset.describe())
print(dataset.CarName.unique())
correlations= dataset.corr()
print(correlations)

#plotting with dataset
sns.histplot(dataset.price)
plt.show()

plt.figure(figsize=(20, 15))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()

#selecting and splitting the dataset
exception = "price"
dataset = dataset[["symboling", "wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginesize", "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg", "price"]]
X = np.array(dataset.drop([exception], 1))
y = np.array(dataset[exception])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#creating the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model.score(X_test, predictions)

#creating new df
new_dataset = pd.DataFrame({'New predictions': predictions})
print(new_dataset.head())
