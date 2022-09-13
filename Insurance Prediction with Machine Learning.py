#Insurance Prediction with Machine Learning
#https://thecleverprogrammer.com/2021/09/03/insurance-prediction-with-machine-learning/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/TravelInsurancePrediction.csv')
print(dataset.head())
dataset.isnull().sum()

#removingg the unnamed column
dataset.drop(columns=["Unnamed: 0"], inplace=True)

#mapping the dataset
dataset["TravelInsurance"] = dataset["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})
dataset["GraduateOrNot"] = dataset["GraduateOrNot"].map({"No": 0, "Yes": 1})
dataset["FrequentFlyer"] = dataset["FrequentFlyer"].map({"No": 0, "Yes": 1})
dataset["EverTravelledAbroad"] = dataset["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
print(dataset.head())

#plotting
sns.histplot(x='Age', data=dataset, hue='TravelInsurance')
plt.title('Factors Affecting Purchase of Travel Insurance: Age')
plt.show()

sns.histplot(x='Employment Type', data=dataset, hue='TravelInsurance')
plt.title('Factors Affecting Purchase of Travel Insurance: Employment Type')
plt.show()

sns.histplot(x='AnnualIncome', data=dataset, hue='TravelInsurance' )
plt.title('Factors Affecting Purchase of Travel Insurance: AnnualIncome')
plt.show()

#selecting and splitting the dataset
X = np.array(dataset[["Age", "GraduateOrNot", "AnnualIncome", "FamilyMembers", "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad"]])
y = np.array(dataset[["TravelInsurance"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model creation and fitting
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#making predictions
predictions = model.predict(X_test)
pred_dataset = pd.DataFrame({'Predicted values': predictions})
print(pred_dataset.head())
