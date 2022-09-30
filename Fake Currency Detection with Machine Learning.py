#Fake Currency Detection with Machine Learning
#https://thecleverprogrammer.com/2020/09/29/fake-currency-detection-with-machine-learning/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#reading and viewing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/data_banknote_authentication.txt', header=None)
print(dataset.head())
dataset.isnull().sum()
dataset.describe()
dataset.info()

#assigning columns
dataset.columns = ['var', 'skew', 'curt', 'entr', 'auth']
dataset.head()

#finding the correlation
correlation = dataset.corr()
print(correlation)

#visualization
sns.pairplot(dataset, hue='auth')
plt.show()

plt.figure(figsize=(8,6))
plt.title('Distribution of Target', size=18)
sns.countplot(x=dataset['auth'])
target_count = dataset.auth.value_counts()
plt.annotate(text=target_count[0], xy=(-0.04,10+target_count[0]), size=14)
plt.annotate(text=target_count[1], xy=(0.96,10+target_count[1]), size=14)
plt.ylim(0,900)
plt.show()

#processing dataset
nb_to_delete = target_count[0] - target_count[1]
dataset = dataset.sample(frac=1, random_state=42).sort_values(by='auth')
dataset = dataset[nb_to_delete:]
print(dataset['auth'].value_counts())

#selecting and splitting the dataset
X = dataset.loc[:, dataset.columns != 'auth']
y = dataset.loc[:, dataset.columns == 'auth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#model creation and fitting using standardscaler
scalar_model = StandardScaler()
scalar_model.fit(X_train)
X_train = scalar_model.transform(X_train)
X_test = scalar_model.transform(X_test)

#model creation and fitting using logisticreg
log_reg = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
log_reg.fit(X_train, y_train.values.ravel())

#making predictions
predictions = np.array(log_reg.predict(X_test))
conf_mat = pd.DataFrame(confusion_matrix(y_test, predictions),
                        columns=["Pred.Negative", "Pred.Positive"],
                        index=['Act.Negative', "Act.Positive"])
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
accuracy = round((tn+tp)/(tn+fp+fn+tp), 4)
print(conf_mat)
print(f'\n Accuracy = {round(100*accuracy, 2)}%')


new_banknote = np.array([4.5, -8.1, 2.4, 1.4], ndmin=2)
new_banknote = scalar_model.transform(new_banknote)
print(f'Prediction:  Class{log_reg.predict(new_banknote)[0]}')
print(f'Probability [0/1]:  {log_reg.predict_proba(new_banknote)[0]}')
