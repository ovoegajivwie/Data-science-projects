#Health Insurance Premium Prediction using Python
#https://thecleverprogrammer.com/2021/10/26/health-insurance-premium-prediction-with-machine-learning/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#reading and viewing the dataset
#https://www.kaggle.com/shivadumnawar/health-insurance-dataset/download
hipp = pd.read_csv('Health_insurance.csv')
print(hipp.head())

#checking for missing values
hipp.isnull().sum()

#visualizing the dataset
sns.histplot(x='sex', data=hipp, hue = "smoker")
plt.title('Numer of smokers')
plt.show()

#mapping our data with string values
hipp["sex"] = hipp["sex"].map({"female": 0, "male": 1})
hipp["smoker"] = hipp["smoker"].map({"no": 0, "yes": 1})
print(hipp.head())

#visualizing the regions
sns.histplot(x='region', data=hipp)
plt.show()

#Finding the correlations
print(hipp.corr())

#electing and splitting the dataset
X = np.array(hipp[["age", "sex", "bmi", "smoker"]])
y = np.array(hipp["charges"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creatinf and fitting a model
model = RandomForestRegressor()
model.fit(X_train, y_train)

#making predictions
y_pred = model.predict(X_test)
pred_values = pd.DataFrame({"Predicted Premium Amount": y_pred})
print(pred_values.head())
