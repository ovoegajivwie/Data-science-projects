#Stress Detection with Machine Learning
#https://thecleverprogrammer.com/2021/12/20/stress-detection-with-machine-learning/

import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

#reading and viewing dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/stress.csv')
print(dataset.head())
dataset.info()
dataset.describe()
print(dataset.isnull().sum())

#cleaning up text in dataset
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
dataset["text"] = dataset["text"].apply(clean)

#visualization
text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#mapping columns
dataset["label"] = dataset["label"].map({0: "No Stress", 1: "Stress"})
dataset = dataset[["text", "label"]]
print(dataset.head())

#selecting
x = np.array(dataset["text"])
y = np.array(dataset["label"])

#model creation
cv_model = CountVectorizer()
X = cv_model.fit_transform(x)

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=0.2, 
                                                random_state=42)

#model creation and fitting
bn_model = BernoulliNB()
bn_model.fit(X_train, y_train)

#testing
user = input("Enter a Text: ")
dataset = cv.transform([user]).toarray()
output = model.predict(dataset)
print(output)
