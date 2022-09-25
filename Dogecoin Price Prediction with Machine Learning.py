#Dogecoin Price Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/05/25/dogecoin-price-prediction-with-machine-learning/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance
import datetime
from datetime import date, timedelta
from autots import AutoTS

#reading todays date
today = date.today()

#getting the start and end dates
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

#reading and viewing the dataset
dataset = yfinance.download('DOGE-USD', start=start_date, end=end_date, progress=False)
dataset.dropna()
correlation = dataset.corr()
print(dataset.head())

#indexing
dataset["Date"] = dataset.index

#selecting
dataset = dataset[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
dataset.reset_index(drop=True, inplace=True)
print(dataset.head())

#setting plotstyle
sns.set()
plt.style.use('seaborn-whitegrid')

#making plots
plt.figure(figsize=(10, 4))
plt.title("DogeCoin Price INR")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(dataset["Close"])
plt.show()

#model creation and fitting
model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(dataset, date_col='Date', value_col='Close', id_col=None)

#making predictions
prediction = model.predict()
forecast = prediction.forecast
print("DogeCoin Price Prediction")
print(forecast)
