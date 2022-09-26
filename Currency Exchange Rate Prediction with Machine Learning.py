#Currency Exchange Rate Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/05/22/currency-exchange-rate-prediction-with-machine-learning/
#I made use of a more recent dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime, date, timedelta


#getting the dates
today = date.today()
end_date = today
d2 = end_date - timedelta(days=365)
start_date = d2

#fetching the dataset
dataset = yfinance.download('INR=X', start = start_date, end=end_date, progress=False)

#indexing
dataset["Date"] = dataset.index
dataset = dataset[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
dataset.reset_index(drop=True, inplace=True)
print(dataset.head())

#setting plot style abd size
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 4))

#plotting
plt.plot(dataset["Close"])
plt.title("INR - USD Exchange Rate")
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()

#finding the correlation
correlation = dataset.corr()
print(correlation)

#plotting with correlation
sns.heatmap(dataset.corr())
plt.show()

#selecting and reshaping the dataset
X = dataset[["Open", "High", "Low"]]
y = dataset["Close"]
X = X.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

#splitting the daaset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model creation and fitting
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#making predictions
predictions = model.predict(X_test)

#converting predictions into a new dataset
data = pd.DataFrame({"Predicted Rate": predictions.flatten()})
print(data.head())
