#Online Payments Fraud Detection with Machine Learning
#https://thecleverprogrammer.com/2022/02/22/online-payments-fraud-detection-with-machine-learning/
#https://www.kaggle.com/ealaxi/paysim1/download

#making necessary imports
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#reading and viewing datasetset
dataset = pd.read_csv("credit card datasetset.csv")
print(dataset.head())
print(dataset.isnull().sum())
print(dataset.info())
print(dataset.describe())
print(dataset.type.value_counts())
correlation = dataset.corr()
print(correlation)
print(correlation["isFraud"].sort_values(ascending=False))

#making plots
dataset_type = dataset["type"].value_counts()
dataset_transactions = dataset_type.index
dataset_quantity = dataset_type.values

figure = px.pie(dataset, 
             values = dataset_quantity, 
             names = dataset_transactions,hole = 0.5, 
             title = "Distribution of Transaction Type")
figure.show()

#mapping
dataset["type"] = dataset["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
dataset["isFraud"] = dataset["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(dataset.head())

#splitting datasetset
X = np.array(dataset[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(dataset[["isFraud"]])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

#model creation
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train, y_train)
print(dtc_model.score(X_test, y_test))

#making predictions
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(dtc_model.predict(features))
