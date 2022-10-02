#Ridge and Lasso Regression with Python
#https://thecleverprogrammer.com/2020/10/09/ridge-and-lasso-regression-with-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/Advertising.csv')
dataset.head()
dataset.isnull().sum()
dataset.info()
dataset.describe()
dataset.drop(["Unnamed: 0"], axis=1, inplace=True)

#getting the correlation
correlation = dataset.corr()
print(correlation)

#visualizing the amount spent on advertizement on TV vs unit sold
plt.scatter(x='TV', y='Sales', data=dataset)
plt.title('Money spent on TV ads ($)')
plt.show()

#visualizing the amount spent on advertizement on TV vs unit sold
plt.scatter(x='Radio', y='Sales', data=dataset)
plt.title('Money spent on Radio ads ($)')
plt.show()

#visualizing the amount spent on advertizement on TV vs unit sold
plt.scatter(x='Newspaper', y='Sales', data=dataset)
plt.title('Money spent on Newspaper ads ($)')
plt.show()

#selecting the data
X = dataset.drop(["Sales"], axis=1)
y = dataset["Sales"].values.reshape(-1,1)

#Linear Regression
linear_model = LinearRegression()
MSE = cross_val_score(linear_model, X, y, scoring="neg_mean_squared_error", cv=5)
#getting MSE mean
mean_MSE = np.mean(MSE)
print(mean_MSE)

#Ridge Regression
ridge_model = Ridge()
parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regression = GridSearchCV(ridge_model, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regression.fit(X, y)
print(ridge_regression.best_params_)
print(ridge_regression.best_score_)

#Lasso Regression
lasso_model = Lasso()
parameters = {"alpha":[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regression = GridSearchCV(lasso_model, parameters, scoring='neg_mean_squared_error', cv=5)
lasso_regression.fit(X, y)
print(lasso_regression.best_params_)
print(lasso_regression.best_score_)
