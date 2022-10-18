import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_rows', 500)

#train dataset
train_dataset = pd.read_csv('kaggle3_train.csv')
#print(train_dataset.head())

#test dataset
test_dataset = pd.read_csv('kaggle3_test.csv')
#print(test_dataset.head())

#filling nan columns(train)
train_dataset['LotFrontage'] = train_dataset['LotFrontage'].fillna(train_dataset['LotFrontage'].mean())
train_dataset.MasVnrType.fillna(method='ffill', inplace = True)
train_dataset.MasVnrArea.fillna(method='ffill', inplace = True)
train_dataset.BsmtQual.fillna(method='ffill', inplace = True)
train_dataset.BsmtCond.fillna(method='ffill', inplace = True)
train_dataset.BsmtExposure.fillna(method='ffill', inplace = True)
train_dataset.BsmtFinType1.fillna(method='ffill', inplace = True)
train_dataset.BsmtFinType2.fillna(method='ffill', inplace = True)
train_dataset.Electrical.fillna(method='ffill', inplace = True)
train_dataset.GarageType.fillna(method='ffill', inplace = True)
train_dataset.GarageYrBlt.fillna(method='ffill', inplace = True)
train_dataset.GarageFinish.fillna(method='ffill', inplace = True)
train_dataset.GarageQual.fillna(method='ffill', inplace = True)
train_dataset.GarageCond.fillna(method='ffill', inplace = True)

#filling na columns(test)
test_dataset.MSZoning.fillna(method='ffill', inplace = True)
test_dataset['LotFrontage'] = test_dataset['LotFrontage'].fillna(test_dataset['LotFrontage'].mean())
test_dataset.Utilities.fillna(method='ffill', inplace = True)
test_dataset.Exterior1st.fillna(method='ffill', inplace = True)
test_dataset.Exterior2nd.fillna(method='ffill', inplace = True)
test_dataset.MasVnrType.fillna(method='ffill', inplace = True)
test_dataset['MasVnrArea'] = test_dataset['MasVnrArea'].fillna(test_dataset['MasVnrArea'].mean())
test_dataset.BsmtQual.fillna(method='ffill', inplace = True)
test_dataset.BsmtCond.fillna(method='ffill', inplace = True)
test_dataset.BsmtExposure.fillna(method='ffill', inplace = True)
test_dataset.BsmtFinType1.fillna(method='ffill', inplace = True)
test_dataset.BsmtFinSF1.fillna(method='ffill', inplace = True)
test_dataset.BsmtFinType2.fillna(method='ffill', inplace = True)
test_dataset.BsmtFinSF2.fillna(method='ffill', inplace = True)
test_dataset.BsmtUnfSF.fillna(method='ffill', inplace = True)
test_dataset.TotalBsmtSF.fillna(method='ffill', inplace = True)
test_dataset.BsmtFullBath.fillna(method='ffill', inplace = True)
test_dataset.BsmtHalfBath.fillna(method='ffill', inplace = True)
test_dataset.KitchenQual.fillna(method='ffill', inplace = True)
test_dataset.Functional.fillna(method='ffill', inplace = True)
test_dataset.GarageType.fillna(method='ffill', inplace = True)
test_dataset.GarageYrBlt.fillna(method='ffill', inplace = True)
test_dataset.GarageFinish.fillna(method='ffill', inplace = True)
test_dataset.GarageQual.fillna(method='ffill', inplace = True)
test_dataset.GarageCond.fillna(method='ffill', inplace = True)
test_dataset.GarageCars.fillna(method='ffill', inplace = True)
test_dataset.GarageArea.fillna(method='ffill', inplace = True)
test_dataset.SaleType.fillna(method='ffill', inplace = True)

#selecting(train)
train_dataset = train_dataset.drop(['Id','Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'GarageType', 'SaleType', 'SaleCondition'], axis=1)
train_data = train_dataset[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
       '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea',
       'WoodDeckSF', 'OpenPorchSF', 'MoSold', 'YrSold']]
print(train_dataset.isnull().sum())
print(train_dataset.info())

#selecting(test)
test_dataset = test_dataset.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'GarageType', 'SaleType', 'SaleCondition'], axis=1)
test_data = test_dataset[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
       '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea',
       'WoodDeckSF', 'OpenPorchSF', 'MoSold', 'YrSold']]
print(test_dataset.isnull().sum())
print(test_dataset.info())

#visualization(train)
correlation = train_dataset.corr()
print(correlation)
plt.figure(figsize=(25, 25))
sns.heatmap(correlation, annot=True)
plt.show()

#visualization(test)
correlation = test_dataset.corr()
print(correlation)
plt.figure(figsize=(25, 25))
sns.heatmap(correlation, annot=True)
plt.show()

#unique vals(train)
print(train_dataset.apply(lambda col: col.unique()))

#unique vals(test)
print(test_dataset.apply(lambda col: col.unique()))

#mapping(train)

train_data["BsmtFinType1"] = train_data["BsmtFinType1"].map({"GLQ": 0, "ALQ": 1, 'Unf': 2, "Rec": 3, "BLQ": 4, "LwQ": 5})
print(train_data.head())

#mapping(test)

test_data["BsmtFinType1"] = test_data["BsmtFinType1"].map({"GLQ": 0, "ALQ": 1, 'Unf': 2, "Rec": 3, "BLQ": 4, "LwQ": 5})
print(test_data.head())

#selecting cols
train_x = train_data
test_x = test_data

y = train_dataset['SalePrice']

#splitting
X_train, X_test, y_train, y_test = train_test_split(train_x, y, test_size=0.2, random_state=42)

#model creation
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print('LinearRegression:', lr_model.score(X_train, y_train))

#prediction
lr_pred = lr_model.predict(X_test)
print("LinearRegression pred:",lr_pred)

#THE KEY TO HOW I GOT MY SELECTED VALUES
#rfe
#rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=20, step=10, verbose=1)
#rfe.fit(X_train, y_train)
#print(train_x.columns[rfe.support_])

#predicting with test dataset
lr_test_dataset_pred = lr_model.predict(test_x)
print("LinearRegression pred:",lr_test_dataset_pred)

#saving as csv
#lr
lr_result = pd.DataFrame(test_dataset['Id'])
print(lr_result.head(10))
lr_result.insert(1, "SalePrice", lr_test_dataset_pred, True)
lr_result["SalePrice"] = pd.to_numeric(lr_result["SalePrice"], downcast="integer")
print(lr_result)
lr_result.to_csv("lr_selected_kaggle3_output.csv", index=False)
