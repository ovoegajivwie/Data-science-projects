#Contact Tracing with Machine Learning
#https://thecleverprogrammer.com/2020/08/20/contact-tracing-with-machine-learning/
#changed a few things from the original, thats why it varied(e.g: the model metric)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import datetime as dt
from sklearn.cluster import DBSCAN

#reading and viewing dataset
dataset = pd.read_json('https://raw.githubusercontent.com/amankharwal/Website-data/master/livedata.json')
dataset.head()
dataset.isnull().sum()
dataset.describe()
dataset.info()
dataset.shape

#correlation
correlation = dataset.corr()
print(correlation)

#visualization
plt.figure(figsize=(8,5))
sns.scatterplot(x='latitude', y='longitude', data=dataset, hue='id')
plt.legend(bbox_to_anchor= [1, 0.8])
plt.show()

X= dataset[['latitude']]
y=dataset[['longitude']]

epsilon = 0.0018288 # a radial distance of 6 feet in kilometers
model = DBSCAN(eps=epsilon, min_samples=2, metric='euclidean').fit(X, y)
dataset['cluster'] = model.labels_.tolist()


#model for performing tracing
def get_infected_names(input_name):

    input_name_clusters = []
    for i in range(len(dataset)):
        if dataset['id'][i] == input_name:
            if dataset['cluster'][i] in input_name_clusters:
                pass
            else:
                input_name_clusters.append(dataset['cluster'][i])
    
    infected_names = []
    for cluster in input_name_clusters:
        if cluster != -1:
            ids_in_cluster = dataset.loc[dataset['cluster'] == cluster, 'id']
            for i in range(len(ids_in_cluster)):
                member_id = ids_in_cluster.iloc[i]
                if (member_id not in infected_names) and (member_id != input_name):
                    infected_names.append(member_id)
                else:
                    pass
    return infected_names

#using model to generate clusters
labels = model.labels_
fig = plt.figure(figsize=(12,10))
sns.scatterplot(x=dataset['latitude'], y=dataset['longitude'], hue = ['cluster-{}'.format(x) for x in labels])
plt.legend(bbox_to_anchor = [1, 1])
plt.show()

print(get_infected_names("Erin"))
print(get_infected_names("Ivan"))
print(get_infected_names("Bob"))
print(get_infected_names("Heidi"))
