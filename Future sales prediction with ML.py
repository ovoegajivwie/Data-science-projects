#Using python and ML to predictict the amount spent on advertisement vs unit of item sold

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

#reading and viewing the data
future_sales_data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(future_sales_data.head())

#checking for null columns
future_sales_data.isnull().sum()

#visualizing the amount spent on advertizement on TV vs unit sold
#plt.scatter(x='TV', y='Sales', data=future_sales_data)
sns.regplot(x='TV', y='Sales', data=future_sales_data)
plt.show()

#visualizing the amount spent on advertizement on Newspaper vs unit sold
#plt.scatter(x='Newspaper', y='Sales', data=future_sales_data)
sns.regplot(x='Newspaper', y='Sales', data=future_sales_data)
plt.show()

#visualizing the amount spent on advertizement on TV vs unit sold
#plt.scatter(x='Radio', y='Sales', data=future_sales_data)
sns.regplot(x='Radio', y='Sales', data=future_sales_data)
plt.show()

#looking at the correlation of the columns with the sales columns
future_sales_data_correlation = future_sales_data.corr()
print(future_sales_data_correlation["Sales"].sort_values(ascending=False))

#training a machine learning model using train_test_split
X = np.array(future_sales_data[['TV', 'Radio', 'Newspaper']])
y = np.array(future_sales_data['Sales'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and fitting the model
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#inputing values to test our model
features = np.array([[151.5, 41.3, 58.5]])
print(model.predict(features))
