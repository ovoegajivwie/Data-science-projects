#Predict US Elections with Python
#https://thecleverprogrammer.com/2020/10/01/predict-us-elections-with-python/
#https://github.com/amankharwal/Website-data/blob/master/US%20Election%20using%20twitter%20sentiment.rar

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px

#reading and viewing dataset
trump_reviews_dataset = pd.read_csv("Trumpall2.csv")
biden_reviews_dataset = pd.read_csv("Bidenall2.csv")

print(trump_reviews_dataset.head())
print(biden_reviews_dataset.head())

textblob1 = TextBlob(trump_reviews_dataset["text"][10])
print("Trump :",textblob1.sentiment)
textblob2 = TextBlob(biden_reviews_dataset["text"][500])
print("Biden :",textblob2.sentiment)


def find_pol(review):
    return TextBlob(review).sentiment.polarity
trump_reviews_dataset["Sentiment Polarity"] = trump_reviews_dataset["text"].apply(find_pol)
print(trump_reviews_dataset.tail())

biden_reviews_dataset["Sentiment Polarity"] = biden_reviews_dataset["text"].apply(find_pol)
print(biden_reviews_dataset.tail())

trump_reviews_dataset["Expression Label"] = np.where(trump_reviews_dataset["Sentiment Polarity"]>0, "positive", "negative")
trump_reviews_dataset["Expression Label"][trump_reviews_dataset["Sentiment Polarity"]==0]="Neutral"
print(trump_reviews_dataset.tail())

biden_reviews_dataset["Expression Label"] = np.where(biden_reviews_dataset["Sentiment Polarity"]>0, "positive", "negative")
biden_reviews_dataset["Expression Label"][trump_reviews_dataset["Sentiment Polarity"]==0]="Neutral"
print(biden_reviews_dataset.tail())


trump_reviews = trump_reviews_dataset[trump_reviews_dataset['Sentiment Polarity'] == 0.0000]
print(trump_reviews.shape)

trump_cond=trump_reviews_dataset['Sentiment Polarity'].isin(trump_reviews['Sentiment Polarity'])
trump_reviews_dataset.drop(trump_reviews_dataset[trump_cond].index, inplace = True)
print(trump_reviews_dataset.shape)

biden_reviews = biden_reviews_dataset[biden_reviews_dataset['Sentiment Polarity'] == 0.0000]
print(biden_reviews.shape)

biden_cond=biden_reviews_dataset['Sentiment Polarity'].isin(trump_reviews['Sentiment Polarity'])
biden_reviews_dataset.drop(biden_reviews_dataset[biden_cond].index, inplace = True)
print(biden_reviews_dataset.shape)

# Donald Trump
np.random.seed(10)
remove_n =324
drop_indices = np.random.choice(trump_reviews_dataset.index, remove_n, replace=False)
df_subset_trump = trump_reviews_dataset.drop(drop_indices)
print(df_subset_trump.shape)
# Joe Biden
np.random.seed(10)
remove_n =31
drop_indices = np.random.choice(biden_reviews_dataset.index, remove_n, replace=False)
df_subset_biden = biden_reviews_dataset.drop(drop_indices)
print(df_subset_biden.shape)


count_1 = df_subset_trump.groupby('Expression Label').count()
print(count_1)

negative_per1 = (count_1['Sentiment Polarity'][0]/1000)*10
positive_per1 = (count_1['Sentiment Polarity'][1]/1000)*100

count_2 = df_subset_biden.groupby('Expression Label').count()
print(count_2)

negative_per2 = (count_2['Sentiment Polarity'][0]/1000)*100
positive_per2 = (count_2['Sentiment Polarity'][1]/1000)*100

Politicians = ['Joe Biden', 'Donald Trump']
lis_pos = [positive_per1, positive_per2]
lis_neg = [negative_per1, negative_per2]

fig = go.Figure(data=[
    go.Bar(name='Positive', x=Politicians, y=lis_pos),
    go.Bar(name='Negative', x=Politicians, y=lis_neg)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()
