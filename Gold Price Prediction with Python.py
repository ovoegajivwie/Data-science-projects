#Gold Price Prediction with Python
#https://thecleverprogrammer.com/2020/12/23/gold-price-prediction-with-python/

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/gold_price.csv', parse_dates=True, index_col='Date')
dataset.head()
dataset.isnull().sum()
dataset.describe()
dataset.info()

#finding the correlation
correlation = dataset.corr()
print(correlation)

#creating new columns
dataset['Return'] = dataset['USD (PM)'].pct_change() * 100
dataset['Lagged_Return'] = dataset.Return.shift()
dataset = dataset.dropna()

#selection
train = dataset['2001':'2018']
test = dataset['2019']
# Create train and test sets for dependent and independent variables
X_train = train["Lagged_Return"].to_frame()
y_train = train["Return"]
X_test = test["Lagged_Return"].to_frame()
y_test = test["Return"]

#model creation and fitting
model = LinearRegression()
model.fit(X_train, y_train)

#making predictions
predictions = model.predict(X_test)

#visualizing the predicted results
out_of_sample_results = y_test.to_frame()
# Add a column of "out-of-sample" predictions to that dataframe:  
out_of_sample_results["Out-of-Sample Predictions"] = model.predict(X_test)
out_of_sample_results.plot(subplots=True, title='Gold prices, USD')
plt.show()
