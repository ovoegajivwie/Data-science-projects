#Cryptocurrency Price Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/12/27/cryptocurrency-price-prediction-with-machine-learning/

import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from autots import AutoTS

#reading todays date
today = date.today()

#
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

#fetching the data
dataset = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)

#indexing
dataset["Date"] = dataset.index
#selecting
dataset = dataset[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
dataset.reset_index(drop=True, inplace=True)
print(dataset.head())

#visualizing the dataset
figure = go.Figure(data=[go.Candlestick(x=dataset["Date"],
                                        open=dataset["Open"], 
                                        high=dataset["High"],
                                        low=dataset["Low"], 
                                        close=dataset["Close"])])
figure.update_layout(title = "Bitcoin Price Analysis", 
                     xaxis_rangeslider_visible=False)
figure.show()

#getting the correlation
correlation = dataset.corr()
print(correlation["Close"].sort_values(ascending=False))

#model creation and fitting
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(dataset, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)
