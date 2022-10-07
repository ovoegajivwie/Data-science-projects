#SMS Spam Detection with Machine Learning
#https://thecleverprogrammer.com/2020/06/12/sms-spam-detection-with-machine-learning/
#link to dataset(https://thecleverprogrammer.com/wp-content/uploads/2020/06/spam.csv)

import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#reading and viewing the dataset
dataset = pd.read_csv('spam.csv', encoding='latin-1')
dataset.head()
dataset.isnull().sum()
dataset.describe()
dataset.info()
print(dataset.shape)

#correlation
correlation = dataset.corr()
print(correlation)

#dropping redundant cols
dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset = dataset.rename(columns={"v1":"label", "v2":"sms"})
dataset.head()

#analysing the dataset
print(len(dataset))
dataset.label.value_counts()
dataset['length'] = dataset['sms'].apply(len)

#visualization
plt.hist(dataset.length, bins=50)
plt.show()

dataset.hist(column='length', by='label', bins=50, figsize=(10,4))
plt.show()

#mapping cols
dataset.loc[:,'label'] = dataset.label.map({'ham':0, 'spam':1})
print(dataset.shape)
dataset.head()

#selectig and splitting
X_train, X_test, y_train, y_test = train_test_split(dataset['sms'], 
                                                    dataset['label'],test_size=0.2, 
                                                    random_state=42)

#model creation and fitting
model = CountVectorizer()
training_data = model.fit_transform(X_train)

# Transform testing data and return the matrix. 
testing_data = model.transform(X_test)

#model creation and fitting
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(training_data, y_train)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#making predictions
predictions = naive_bayes_model.predict(testing_data)

#model evaluation
print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))
print('Precision score: {}'.format(precision_score(y_test, predictions)))
print('Recall score: {}'.format(recall_score(y_test, predictions)))
print('F1 score: {}'.format(f1_score(y_test, predictions)))
