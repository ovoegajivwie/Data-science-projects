#product demand prediction with ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#reading and veiwing the data
demand = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
print(demand.head())

#checking for null values
print(demand.isnull().sum())

#dropping null column
demand = demand.dropna()
print(demand.isnull().sum())

#visualizing the data
sns.scatterplot(x="Units Sold", y="Total Price",
                 size='Units Sold', data=demand)
plt.show()

#getting the corr
correlations = demand.corr(method='pearson')
print(correlations)

#visualizing with heatmap
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()

#splitting the data
X = demand[["Total Price", "Base Price"]]
y = demand["Units Sold"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating and fitting the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

#making predictions
features = np.array([[99.00, 111.00]])
model.predict(features)
