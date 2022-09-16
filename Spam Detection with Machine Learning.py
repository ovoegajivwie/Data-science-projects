#Spam Detection with Machine Learning
#https://thecleverprogrammer.com/2021/06/27/spam-detection-with-machine-learning/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv', encoding= 'latin-1')
print(dataset.head())

dataset = dataset[["class", "message"]]

#selecting and splitting the dataset
x = np.array(dataset["message"])
y = np.array(dataset["class"])

#creating our model
cv = CountVectorizer()
X = cv.fit_transform(x)

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model creation and fitting
clf = MultinomialNB()
clf.fit(X_train, y_train)


test = input('Enter a message:')
new_dataset = cv.transform([test]).toarray()
print(clf.predict(new_dataset))
