#Heart Disease Prediction with Machine Learning
#https://thecleverprogrammer.com/2020/05/20/heart-disease-prediction-with-machine-learning/
#https://thecleverprogrammer.com/wp-content/uploads/2020/05/heart.csv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#reading and viewing the dataset
dataset = pd.read_csv('heart.csv')
print(dataset.head())
dataset.isnull().sum()
dataset.info()
dataset.describe()

#correlation
correlation = dataset.corr()
print(correlation)

#visualization
plt.figure(figsize=(16,16))
sns.heatmap(correlation, annot=True, cmap="RdYlGn")
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='target',data=dataset,palette='RdBu_r')
plt.show()

df = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs','restecg', 'exang', 'slope', 'ca', 'thal'])
df.head()

#model creation
standardScaler = StandardScaler()
selected_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[selected_cols] = standardScaler.fit_transform(df[selected_cols])
df.head()

#selection
X = df.drop(['target'], axis = 1)
y = df['target']

#creating and empty list to hold the scores
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score = cross_val_score(knn_classifier, X, y, cv=10)
    knn_scores.append(score.mean())

plt.figure(figsize=(16,16))
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()

#knn model
knn_classifier = KNeighborsClassifier(n_neighbors = 12)
knn_score=cross_val_score(knn_classifier, X, y, cv=10)
print(knn_score.mean())

#rfc model
rfc_model= RandomForestClassifier(n_estimators=10)
rfc_score=cross_val_score(rfc_model, X, y, cv=10)
print(rfc_score.mean())
