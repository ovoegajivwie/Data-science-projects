#Social Media Followers Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/08/09/social-media-followers-prediction-with-machine-learning/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autots import AutoTS

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/stats.csv')
dataset.head()
dataset.drop(dataset.tail(1).index, inplace=True)
dataset.isnull().sum()
dataset.describe()

#visualizing the dataset
plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Number of Followers I Gained Every Month")
sns.barplot(x="followers_gained", y="period_end", data=dataset)
plt.show()

plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Total Followers At The End of Every Month")
sns.barplot(x="followers_total", y="period_end", data=dataset)
plt.show()

plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
plt.title("Total Views Every Month")
sns.barplot(x="views", y="period_end", data=dataset)
plt.show()

#model creation 
model = AutoTS(forecast_length=4, frequency='infer', ensemble='simple')
model = model.fit(dataset, date_col='period_end', value_col='followers_gained', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)
