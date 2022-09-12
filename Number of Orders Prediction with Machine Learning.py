#Number of Orders Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/09/27/number-of-orders-prediction-with-machine-learning/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as ltb

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/supplement.csv')
print(dataset.head())
dataset.isnull().sum()
dataset.info()
dataset.describe()

#visualizing the data
sns.histplot(x='Store_Type', data=dataset)
plt.show()
sns.histplot(x='Location_Type', data=dataset)
plt.show()
sns.histplot(x='Discount', data=dataset)
plt.show()
sns.histplot(x='Holiday', data=dataset)
plt.show()

#mapping the dataset
dataset["Discount"] = dataset["Discount"].map({"No": 0, "Yes": 1})
dataset["Store_Type"] = dataset["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4})
dataset["Location_Type"] = dataset["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5})

#selecting and splitting the dataset
X = np.array(dataset[["Store_Type", "Location_Type", "Holiday", "Discount"]])
y = np.array(dataset["#Order"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#creating the model
model = ltb.LGBMRegressor()
model.fit(X_train, y_train)

#making predictions
y_pred = model.predict(X_test)

#creating a dataframe from the result
new_dataset = pd.DataFrame({"Predicted Orders": y_pred})
print(new_dataset.head())
