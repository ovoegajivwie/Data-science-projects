#waiter tip prediction with ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#reading and viewing the data
tips = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv')
print(tips.head(n=5))

#visualizing tip given to waiter according to the total bill paid, gender and time
sns.scatterplot(x='total_bill', y='tip', hue='day', size='size', alpha=0.5, data=tips)
plt.show()
sns.scatterplot(x='total_bill', y='tip', hue='sex', size='size', alpha=0.5, data=tips)
plt.show()
sns.scatterplot(x='total_bill', y='tip', hue='time', size='size', alpha=0.5, data=tips)
plt.show()

#visualizing which day the most tip was given
sns.catplot(x='day', y='tip', data=tips)
plt.show()

#visualizing which gender gave the most tip
sns.catplot(x='sex', y='tip', data=tips)
plt.show()

#visualizing if a smoker tips or not
sns.catplot(x='smoker', y='tip', data=tips)
plt.show()

#transform categorical data into numrical variables
tips["sex"] = tips["sex"].map({"Female": 0, "Male": 1})
tips["smoker"] = tips["smoker"].map({"No": 0, "Yes": 1})
tips["day"] = tips["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
tips["time"] = tips["time"].map({"Lunch": 0, "Dinner": 1})
print(tips.head())

#splitting the data using train_test_split
X = np.array(tips[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
y = np.array(tips["tip"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating the model
model = LinearRegression()
model.fit(X_train, y_train)

#predicting
features = np.array([[10.30, 1, 0, 3, 1, 3]])
model.predict(features)
