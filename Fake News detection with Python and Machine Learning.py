#Detecting Fake News with Python and Machine Learning
#https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/
#link to dataset(https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#reading and viewing dataset
dataset = pd.read_csv('news.csv')
dataset.head()
dataset.isnull().sum()
dataset.describe()
dataset.info()
dataset.shape

labels = dataset.label
labels.head()

#selecting and splitting dataset
X = dataset.text
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model creation and fitting
tfidf_vectorizer_model = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer_model.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer_model.transform(X_test)

#model creation and fitting
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

#making predictions
predictions = pac.predict(tfidf_test)
score = accuracy_score(y_test, predictions)
print(f'Accuracy: {round(score*100,2)}%')

#build confusion matrix
confusion_matrix(y_test, predictions, labels=['FAKE','REAL'])
