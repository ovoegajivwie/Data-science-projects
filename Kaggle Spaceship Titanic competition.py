#the SVC predictions are better and have a higher score on kaggle as at time of submission
#kaggle scores(svm:0.78910, lr:0.78138, knn:0.77367, dtc:0.73579)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics

#train dataset
train_dataset = pd.read_csv('kaggle2_train.csv')
print(train_dataset.isnull().sum())
print(train_dataset.info())
print(train_dataset.describe())
train_dataset.head()

#test_dataset
test_dataset = pd.read_csv('kaggle2_test.csv')
print(test_dataset.isnull().sum())
print(test_dataset.info())
print(test_dataset.describe())
test_dataset.head()

#dealing with missing cols for train dataset
train_dataset.HomePlanet.fillna(method='ffill', inplace = True)
train_dataset.CryoSleep.fillna(method='ffill', inplace = True)
train_dataset.Cabin.fillna(method='ffill', inplace = True)
train_dataset.Destination.fillna(method='ffill', inplace = True)
train_dataset.Age = train_dataset.Age.fillna(train_dataset.Age.mean())
train_dataset.VIP.fillna(method='ffill', inplace = True)
train_dataset.RoomService = train_dataset.RoomService.fillna(train_dataset.RoomService.mean())
train_dataset.FoodCourt = train_dataset.FoodCourt.fillna(train_dataset.FoodCourt.mean())
train_dataset.ShoppingMall = train_dataset.ShoppingMall.fillna(train_dataset.ShoppingMall.mean())
train_dataset.Spa = train_dataset.Spa.fillna(train_dataset.Spa.mean())
train_dataset.VRDeck = train_dataset.VRDeck.fillna(train_dataset.VRDeck.mean())
print(train_dataset.isnull().sum())

#dealing with missing cols for test dataset
test_dataset.HomePlanet.fillna(method='ffill', inplace = True)
test_dataset.CryoSleep.fillna(method='ffill', inplace = True)
test_dataset.Cabin.fillna(method='ffill', inplace = True)
test_dataset.Destination.fillna(method='ffill', inplace = True)
test_dataset.Age = test_dataset.Age.fillna(test_dataset.Age.mean())
test_dataset.VIP.fillna(method='ffill', inplace = True)
test_dataset.RoomService = test_dataset.RoomService.fillna(test_dataset.RoomService.mean())
test_dataset.FoodCourt = test_dataset.FoodCourt.fillna(test_dataset.FoodCourt.mean())
test_dataset.ShoppingMall = test_dataset.ShoppingMall.fillna(test_dataset.ShoppingMall.mean())
test_dataset.Spa = test_dataset.Spa.fillna(test_dataset.Spa.mean())
test_dataset.VRDeck = test_dataset.VRDeck.fillna(test_dataset.VRDeck.mean())
print(test_dataset.isnull().sum())

#finding unique values for train dataset
print(train_dataset.HomePlanet.unique())
print(train_dataset.CryoSleep.unique())
#print(train_dataset.Cabin.unique())
print(train_dataset.Destination.unique())
print(train_dataset.VIP.unique())

#finding unique values for test dataset
print(test_dataset.HomePlanet.unique())
print(test_dataset.CryoSleep.unique())
#print(test_dataset.Cabin.unique())
print(test_dataset.Destination.unique())
print(test_dataset.VIP.unique())

#converting cols/mapping
train_dataset["HomePlanet"] = train_dataset["HomePlanet"].map({"Europa": 0, "Earth": 1, 'Mars': 2})
train_dataset["Destination"] = train_dataset["Destination"].map({"TRAPPIST-1e": 0, "PSO J318.5-22": 1, '55 Cancri e': 2})

#test dataset
test_dataset["HomePlanet"] = test_dataset["HomePlanet"].map({"Europa": 0, "Earth": 1, 'Mars': 2})
test_dataset["Destination"] = test_dataset["Destination"].map({"TRAPPIST-1e": 0, "PSO J318.5-22": 1, '55 Cancri e': 2})

#selecting cols
train_X = train_dataset[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
test_X = test_dataset[['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]

y = train_dataset['Transported']

#splitting
X_train, X_test, y_train, y_test = train_test_split(train_X, y, test_size=0.2, random_state=42)

#model creation and fitting
lr_model = LogisticRegression()
dtc_model = DecisionTreeClassifier()
rfc_model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=1111)
knn_model = KNeighborsClassifier()
svm_model = SVC()

#fitting
lr_model.fit(X_train, y_train)
dtc_model.fit(X_train, y_train)
rfc_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

print('LogisticRegression:', lr_model.score(X_train, y_train))
print('DecisionTreeClassifier:', dtc_model.score(X_train, y_train))
print('RandomForestRegressor:', rfc_model.score(X_train, y_train))
print('KNeighborsClassifier:', knn_model.score(X_train, y_train))
print('svc:', svm_model.score(X_train, y_train))

#cvs
lr_cv_results = cross_val_score(lr_model, train_X, y, cv=5)
dtc_cv_results = cross_val_score(dtc_model, train_X, y, cv=5)
rfc_cv_results = cross_val_score(rfc_model, train_X, y, cv=5)
knn_cv_results = cross_val_score(knn_model, train_X, y, cv=5)
svm_cv_results = cross_val_score(svm_model, train_X, y, cv=5)

print('LogisticRegression:', lr_cv_results)
print('DecisionTreeClassifier:', dtc_cv_results)
print('RandomForestRegressor:', rfc_cv_results)
print('KNeighborsClassifier:', knn_cv_results)
print('svc:', svm_cv_results)

#prediction
lr_pred = lr_model.predict(X_test)
dtc_pred = dtc_model.predict(X_test)
#rfc_pred = rfc_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

print("LogisticRegression pred:",lr_pred)
print("DecisionTreeClassifier pred:",dtc_pred)
#print("RandomForestRegressor pred:",rfc_pred)
print("KNeighborsClassifier pred:",knn_pred)
print("svc pred:",svm_pred)

#accuracy score
print("LogisticRegression Accuracy:",metrics.accuracy_score(y_test, lr_pred))
print("DecisionTreeClassifier Accuracy:",metrics.accuracy_score(y_test, dtc_pred))
#print("RandomForestRegressor Accuracy:",metrics.accuracy_score(y_test, rfc_pred))
print("KNeighborsClassifier Accuracy:",metrics.accuracy_score(y_test, knn_pred))
print("svc Accuracy:",metrics.accuracy_score(y_test, svm_pred))

#predicting with test dataset
lr_test_dataset_pred = lr_model.predict(test_X)
dtc_test_dataset_pred = dtc_model.predict(test_X)
knn_test_dataset_pred = knn_model.predict(test_X)
svm_test_dataset_pred = svm_model.predict(test_X)

print("LogisticRegression pred:",lr_test_dataset_pred)
print("DecisionTreeClassifier pred:",dtc_test_dataset_pred)
print("KNeighborsClassifier pred:",knn_test_dataset_pred)
print("svc pred:",svm_test_dataset_pred)

#saving as csv
#lr
lr_result = pd.DataFrame(test_dataset['PassengerId'])
print(lr_result.head(10))
lr_result.insert(1, "Transported", lr_test_dataset_pred, True)
lr_result["Transported"] = pd.to_numeric(lr_result["Transported"], downcast="integer")
print(lr_result)
lr_result.to_csv("lr_kaggle2_output.csv", index=False)

#dtc
dtc_result = pd.DataFrame(test_dataset['PassengerId'])
print(dtc_result.head(10))
dtc_result.insert(1, "Transported", dtc_test_dataset_pred, True)
dtc_result["Transported"] = pd.to_numeric(dtc_result["Transported"], downcast="integer")
print(dtc_result)
dtc_result.to_csv("dtc_kaggle2_output.csv", index=False)

#knn
knn_result = pd.DataFrame(test_dataset['PassengerId'])
print(knn_result.head(10))
knn_result.insert(1, "Transported", knn_test_dataset_pred, True)
knn_result["Transported"] = pd.to_numeric(knn_result["Transported"], downcast="integer")
print(knn_result)
knn_result.to_csv("knn_kaggle2_output.csv", index=False)

#svm
svm_result = pd.DataFrame(test_dataset['PassengerId'])
print(svm_result.head(10))
svm_result.insert(1, "Transported", svm_test_dataset_pred, True)
svm_result["Transported"] = pd.to_numeric(svm_result["Transported"], downcast="integer")
print(svm_result)
svm_result.to_csv("svm_kaggle2_output.csv", index=False)


train_dataset.head()
#Ended up not using random forest regressor because of its low score, accuracy score and cross_val_score
