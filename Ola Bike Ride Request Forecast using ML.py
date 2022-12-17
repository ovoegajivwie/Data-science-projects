#Ola Bike Ride Request Forecast using ML
#https://www.geeksforgeeks.org/ola-bike-ride-request-forecast-using-ml/
#https://www.kaggle.com/competitions/bike-sharing-demand/data (dataset)

#importing the necessary libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
  
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)

#reading and viewing dataset
dataset = pd.read_csv('ola-train.csv')
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())
print(dataset.isnull().sum())

#seperating the date from the time
parts = dataset["datetime"].str.split(" ", n=2, expand=True)
dataset["date"] = parts[0]
dataset["time"] = parts[1].str[:2].astype('int')
dataset.head()

#seperating the day, month, year from date
parts = dataset["date"].str.split("-", n=3, expand=True)
dataset["year"] = parts[0].astype('int')
dataset["month"] = parts[1].astype('int')
dataset["day"] = parts[2].astype('int')
print(dataset.head())

from datetime import datetime
import calendar

def weekend_or_weekday(year, month, day):

	d = datetime(year, month, day)
	if d.weekday() > 4:
		return 0
	else:
		return 1

dataset['weekday'] = dataset.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
print(dataset.head())

def am_or_pm(x):
	if x > 11:
		return 1
	else:
		return 0

dataset['am_or_pm'] = dataset['time'].apply(am_or_pm)
print(dataset.head())

from datetime import date
import holidays

def is_holiday(x):

	india_holidays = holidays.country_holidays('IN')

	if india_holidays.get(x):
		return 1
	else:
		return 0

dataset['holidays'] = dataset['date'].apply(is_holiday)
print(dataset.head())

#dropping the redundant date cols
dataset.drop(['datetime', 'date'],
		axis=1,
		inplace=True)

#plotting relations
features = ['day', 'time', 'month']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
	plt.subplot(2, 2, i + 1)
	dataset.groupby(col).mean()['count'].plot()
plt.show()

features = ['season', 'weather', 'holidays', 'am_or_pm', 'year', 'weekday']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
	plt.subplot(2, 3, i + 1)
	dataset.groupby(col).mean()['count'].plot.bar()
plt.show()

features = ['temp', 'windspeed']
plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sb.distplot(dataset[col])
plt.show()

features = ['temp', 'windspeed']
plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sb.boxplot(dataset[col])
plt.show()

#checking how much data will be loss due to outliers
num_rows = dataset.shape[0] - dataset[dataset['windspeed']<32].shape[0]
print(f'Number of rows that will be lost if we remove outliers is equal to {num_rows}.')

#more plottings
features = ['humidity', 'casual', 'registered', 'count']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
	plt.subplot(2, 2, i + 1)
	sb.boxplot(dataset[col])
plt.show()

sb.heatmap(dataset.corr() > 0.8,
		annot=True,
		cbar=False)
plt.show()

#removing outliers
dataset.drop(['registered', 'time'], axis=1, inplace=True)
dataset = dataset[(dataset['windspeed'] < 32) & (dataset['humidity'] > 0)]

#selecting and splitting dataset
features = dataset.drop(['count'], axis=1)
target = dataset['count'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.1, random_state=22)
print(X_train.shape)
print(X_test.shape)

#models creation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = [LinearRegression(), XGBRegressor(), Lasso(),
		RandomForestRegressor(), Ridge()]

for i in range(5):
	models[i].fit(X_train, y_train)

	print(f'{models[i]} : ')

	train_pred = models[i].predict(X_train)
	print('Training Error : ', mae(y_train, train_pred))

	test_pred = models[i].predict(X_test)
	print('Validation Error : ', mae(y_test, test_pred))
	print()
