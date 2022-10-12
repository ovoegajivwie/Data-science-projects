import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#reading and viewing dataset
dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
print(dataset.head())
dataset.shape
dataset.info()
dataset.describe()
dataset.head()
dataset.isnull().sum()

#filling
dataset.Age.fillna(method ='ffill', inplace = True)
dataset.Embarked.fillna(method ='ffill', inplace = True)

test_dataset.Age.fillna(method ='ffill', inplace = True)
test_dataset.Embarked.fillna(method ='ffill', inplace = True)
test_dataset.Fare.fillna(method ='ffill', inplace = True)


#mapping
dataset["Sex"] = dataset["Sex"].map({"female": 0, "male": 1})
dataset["Embarked"] = dataset["Embarked"].map({"C": 0, "Q": 1, 'S': 2})

test_dataset["Sex"] = test_dataset["Sex"].map({"female": 0, "male": 1})
test_dataset["Embarked"] = test_dataset["Embarked"].map({"C": 0, "Q": 1, 'S': 2})


#selecting and splitting
test_dataset= test_dataset[['PassengerId','Sex','Age','SibSp','Parch','Pclass','Fare','Embarked']]
test_df = test_dataset[['Sex','Age','SibSp','Parch','Pclass','Fare','Embarked']]

X = dataset[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
train, val = train_test_split(X, test_size=0.2, random_state=42)

train_x = train[['Sex','Age','SibSp','Parch','Pclass','Fare','Embarked']]
train_Label = train[['Survived']]

val_X = val[['Sex','Age','SibSp','Parch','Pclass','Fare','Embarked']]
val_Label = val[['Survived']]

#print(train_x)
model = LogisticRegression()
model.fit(train_x, train_Label)
print(model.score(train_x, train_Label))
y_pred = model.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_Label, y_pred))

test_pred = model.predict(test_df)

result = pd.DataFrame(test_dataset['PassengerId'])
print(result.head(10))

result.insert(1, "Survived", test_pred, True)
result["Survived"] = pd.to_numeric(result["Survived"], downcast="integer")
print(result.head(10))

result.to_csv("output.csv", index=False)
