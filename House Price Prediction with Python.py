#House Price Prediction with Python
#https://thecleverprogrammer.com/2020/12/29/house-price-prediction-with-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')
dataset.head()
dataset.info()
dataset.ocean_proximity.value_counts()

#visualizing the dataset
dataset.hist(bins=50, figsize=(10, 8))
plt.show()

#splitting the dataset
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

#visualization
dataset['income_cat'] = pd.cut(dataset['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
dataset['income_cat'].hist()
plt.show()

#Stratified Sampling on Dataset
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]
print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

#removing the Income_cat attribute added
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
dataset = strat_train_set.copy()

#visualizing the dataset in terms of longitude and latitude
dataset.plot(x='longitude', y='latitude', kind='scatter', alpha=0.4, s=dataset['population']/100, label='population',
figsize=(12, 8), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()
#The graph shows house prices in California where red is expensive, blue is cheap, larger circles indicate areas with a larger population.

#getting the correlation
correlation = dataset.corr()
print(correlation)
print(correlation.median_house_value.sort_values(ascending=False))

#finding the corr by visualization with pandas scatter matrix
features = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(dataset[features], figsize=(12, 10))
plt.show()

#adding columns to dataset and finding the new correlation
dataset["rooms_per_household"] = dataset["total_rooms"]/dataset["households"]
dataset["bedrooms_per_room"] = dataset["total_bedrooms"]/dataset["total_rooms"]
dataset["population_per_household"] = dataset["population"]/dataset["households"]

correlation = dataset.corr()
print(correlation["median_house_value"].sort_values(ascending=False))

#Data Preparation
dataset = strat_train_set.drop("median_house_value", axis=1)
dataset_labels = strat_train_set["median_house_value"].copy()

median = dataset["total_bedrooms"].median()
dataset["total_bedrooms"].fillna(median, inplace=True)

dataset_num = dataset.drop("ocean_proximity", axis=1)

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
#creating a pipeline
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
dataset_num_tr = num_pipeline.fit_transform(dataset_num)

num_attribs = list(dataset_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
dataset_prepared = full_pipeline.fit_transform(dataset)

#creating a linear regression model
model = LinearRegression()
model.fit(dataset_prepared, dataset_labels)

data = dataset.iloc[:5]
labels = dataset_labels.iloc[:5]
data_preparation = full_pipeline.transform(data)
print("Predictions: ", model.predict(data_preparation))
