#Electricity price prediction with ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#reading and viewing the data
electric_bill = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv')
print(electric_bill.head)
print(electric_bill.info())

#converting object columns to int
electric_bill.ForecastWindProduction = pd.to_numeric(electric_bill.ForecastWindProduction, errors= 'coerce')
electric_bill.SystemLoadEA = pd.to_numeric(electric_bill.SystemLoadEA, errors= 'coerce')
electric_bill.SMPEA = pd.to_numeric(electric_bill.SMPEA, errors= 'coerce')
electric_bill.ORKTemperature = pd.to_numeric(electric_bill.ORKTemperature, errors= 'coerce')
electric_bill.ORKWindspeed = pd.to_numeric(electric_bill.ORKWindspeed, errors= 'coerce')
electric_bill.CO2Intensity = pd.to_numeric(electric_bill.CO2Intensity, errors= 'coerce')
electric_bill.ActualWindProduction = pd.to_numeric(electric_bill.ActualWindProduction, errors= 'coerce')
electric_bill.SystemLoadEP2 = pd.to_numeric(electric_bill.SystemLoadEP2, errors= 'coerce')
electric_bill.SMPEP2 = pd.to_numeric(electric_bill.SMPEP2, errors= 'coerce')
print(electric_bill.info())

#checking for null values
electric_bill.isnull().sum()

#dropping null values
electric_bill = electric_bill.dropna()

#find the correlation
correlations = electric_bill.corr(method='pearson')

#visualizing our data
plt.figure(figsize=(16, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()

#selecting and splitting the data
X = electric_bill[["Day", "Month", "ForecastWindProduction", "SystemLoadEA", "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", "ActualWindProduction", "SystemLoadEP2"]]
y = electric_bill["SMPEP2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating a model and fitting the data
model = RandomForestRegressor()
model.fit(X_train, y_train)

#predicting with our model
features = np.array([[15, 9, 321.80, 3388.7, 49.26, 6.0, 11.10, 605.42, 346.00, 2834.00]])
model.predict(features)
