#Email spam Detection with Machine Learning
#https://thecleverprogrammer.com/2020/05/17/email-spam-detection-with-machine-learning/

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Email-spam-detection/master/emails.csv')
dataset.head()
dataset.isnull().sum()
dataset.describe()
dataset.info()
dataset.shape
#dropping duplicates
dataset.drop_duplicates(inplace=True)

# download the stopwords package
#nltk.download("stopwords")

#cleaning text and returning tokens
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean
# to show the tokenization
dataset['text'].head().apply(process)

#converting text to token matrix
message = CountVectorizer(analyzer=process).fit_transform(dataset['text'])

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.2, random_state=42)
print(message.shape)

#model creation and fitting
classifier = MultinomialNB().fit(X_train, y_train)
#making predictions
predictions = classifier.predict(X_train)
print(predictions)
print(y_train.values)

print(classification_report(y_train, predictions))
print("Confusion Matrix: \n", confusion_matrix(y_train, predictions))
print("Accuracy: \n", accuracy_score(y_train, predictions))

#using the test set to make predictions
predictions = classifier.predict(X_test)
print(predictions)
print(y_test.values)

print(classification_report(y_test, predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
print("Accuracy: \n", accuracy_score(y_test, predictions))
