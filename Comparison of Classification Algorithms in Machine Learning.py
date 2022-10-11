#Comparison of Classification Algorithms in Machine Learning
#https://thecleverprogrammer.com/2021/10/02/comparison-of-classification-algorithms-in-machine-learning/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

#reading and viewing the dataset
dataset = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv")
print(dataset.head())
dataset.isnull().sum()
dataset.describe()
dataset.info()
dataset.shape

#selecting and splitting
X = np.array(dataset[["Age", "EstimatedSalary"]])
y = np.array(dataset[["Purchased"]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#models creation
decisiontree_model = DecisionTreeClassifier()
logisticregression_model = LogisticRegression()
knearestclassifier_model = KNeighborsClassifier()
#svm_classifier_model = SVC()
bernoulli_naiveBayes_model = BernoulliNB()
passiveAggressive_model = PassiveAggressiveClassifier()

#models fitting
knearestclassifier_model.fit(X_train, y_train)
decisiontree_model.fit(X_train, y_train)
logisticregression_model.fit(X_train, y_train)
passiveAggressive_model.fit(X_train, y_train)

data1 = {"Classification Algorithms": ["KNN Classifier", "Decision Tree Classifier", 
                                       "Logistic Regression", "Passive Aggressive Classifier"],
      "Score": [knearestclassifier_model.score(X,y), decisiontree_model.score(X, y), 
                logisticregression_model.score(X, y), passiveAggressive_model.score(X,y) ]}
score = pd.DataFrame(data1)
print(score)
