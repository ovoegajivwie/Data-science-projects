#Movie Recommendation System with Machine Learning
#https://thecleverprogrammer.com/2020/05/20/data-science-project-movie-recommendation-system/
#link to dataset
#https://thecleverprogrammer.com/wp-content/uploads/2020/05/tmdb_5000_movies.csv
#https://thecleverprogrammer.com/wp-content/uploads/2020/05/tmdb_5000_credits.csv

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

#reading and viewing the dataset
movies_df = pd.read_csv('movies.csv')
credits_df = pd.read_csv('credits.csv')
print(movies_df.head())
print(credits_df.head())
print(movies_df.isnull().sum())
print(credits_df.isnull().sum())
print("Credits:",credits_df.shape)
print("Movies Dataframe:",movies_df.shape)

#renaming and erging columns
renamed_credits_col = credits_df.rename(index=str, columns={"movie_id": "id"})
movies_df_merge = movies_df.merge(renamed_credits_col, on='id')
print(movies_df_merge.head())

#dropping redundant columns
movies_df_cleaned = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
print(movies_df_cleaned.head())
print(movies_df_cleaned.info())
print(movies_df_cleaned.head(1)['overview'])

#makig a recommendation system model and fitting
model = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')
model_matrix = model.fit_transform(movies_df_cleaned['overview'].values.astype('U'))
print(model_matrix)
print(model_matrix.shape)

#sigmoid model
sigmoid_model = sigmoid_kernel(model_matrix, model_matrix)
print(sigmoid_model[0])

#Reverse mapping of indices and movie titles
indices = pd.Series(movies_df_cleaned.index, index=movies_df_cleaned['original_title']).drop_duplicates()
print(indices)
print(indices['Newlyweds'])
print(sigmoid_model[4799])
print(list(enumerate(sigmoid_model[indices['Newlyweds']])))
print(sorted(list(enumerate(sigmoid_model[indices['Newlyweds']])), key=lambda x: x[1], reverse=True))

#recommendation function
def give_recomendations(title, sig=sigmoid_model):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_df_cleaned['original_title'].iloc[movie_indices]

print(give_recomendations('Avatar'))
