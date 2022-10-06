#Video Game Sales Prediction Model with Python
#https://thecleverprogrammer.com/2021/05/28/video-game-sales-prediction-model-with-python/
#Link to dataset (https://www.kaggle.com/datasets/gregorut/videogamesales?resource=download)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#reading and viewing the dataset
dataset = pd.read_csv('vgsales.csv')
dataset.head()
dataset.isnull().sum()
dataset = dataset.dropna()
dataset.isnull().sum()
dataset.shape
dataset.describe()
dataset.info()

#crrelation
correlation = dataset.corr()
print(correlation)

#data visualization
game = dataset.groupby("Genre")["Global_Sales"].count().head(10)
custom_colors = mpl.colors.Normalize(vmin=min(game), vmax=max(game))
colours = [mpl.cm.PuBu(custom_colors(i)) for i in game]
plt.figure(figsize=(7,7))
plt.pie(game, labels=game.index, colors=colours)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Top 10 Categories of Games Sold", fontsize=20)
plt.show()

sns.heatmap(correlation, cmap="winter_r")
plt.show()

#selecting and fitting dataset
X = dataset[["Rank", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
y = dataset["Global_Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model creation
model = LinearRegression()
model.fit(X_train, y_train)
#making predictions
predictions = model.predict(X_test)
print(predictions)
