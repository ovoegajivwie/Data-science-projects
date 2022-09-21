#Apple Stock Price Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/09/08/apple-stock-price-prediction-with-machine-learning/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime as dt
from datetime import timedelta, date
from autots import AutoTS

#reading and viewing the dataset
today = dt.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = dt.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

dataset = yf.download('AAPL',start=start_date, end=end_date, progress=False)
dataset["Date"] = dataset.index
#dataset = dataset[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
dataset.reset_index(drop=True, inplace=True)

#visualizing the dataset
figure = go.Figure(data=[go.Candlestick(x=dataset["Date"],
                                        open=dataset["Open"], 
                                        high=dataset["High"],
                                        low=dataset["Low"], 
                                        close=dataset["Close"])])
figure.update_layout(title = "Bitcoin Price Analysis", 
                     xaxis_rangeslider_visible=False)
figure.show()

#finding the correlation
correlation = dataset.corr()
print(correlation)

#model creation and fitting
model = AutoTS(forecast_length=5, frequency='infer', ensemble='simple')
model = model.fit(dataset, date_col='Date', value_col='Close', id_col=None)

#making predictions
prediction = model.predict()
forecast = prediction.forecast
print(forecast)
