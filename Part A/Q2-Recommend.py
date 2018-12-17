import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from pandas.io.json import json_normalize
#import random


def predict(ratings,neighborhoods, k):
    
    recommendation_dict = {}
    for index,neighborhood in neighborhoods.iterrows():
        
        ratings_to_consider=ratings[ratings['userId'].isin(neighborhood)]
        ratings_to_consider=ratings_to_consider.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        similarities = cosine_similarity(ratings_to_consider)
        

        userId_to_predict = neighborhood[len(neighborhood)-1]
        
        means = ratings_to_consider.mean(axis=1)
        
        user_mean = means.get(userId_to_predict)

        sum_of_similarities = np.sum(similarities[0,1:])

        
        if sum_of_similarities==0:
            continue

        
        for movie in ratings_to_consider:
            
            recommendation = user_mean
            
            new_index=1
            
            for neighbor in neighborhood[:-1]:
            
                sim_user_neighbor = similarities[0,new_index]
                neighbor_mean = means.get(neighbor)
                
            
                new_index+=1
                rating_for_movie = ratings_to_consider.loc[neighbor,movie]


                recommendation += sim_user_neighbor*(rating_for_movie-neighbor_mean) / sum_of_similarities
            
        
            tup_rec = (movie , recommendation)
        
            if userId_to_predict in recommendation_dict:
                recommendation_dict[userId_to_predict].append(tup_rec)
            else:
                recommendation_dict[userId_to_predict] = [tup_rec]
   
    return recommendation_dict
         
        


ratings = pd.read_csv('data/ratings.csv')

ratings_per_user = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

file = open('data/neighbors-k-100-movilens.data','r')
text = file.read()
text = json.loads(text)

neighbors = pd.DataFrame(columns=json_normalize(text['0']).columns)

for row in text:
    neighbors = neighbors.append(json_normalize(text[row]))
    
#neighbors['UserId']+=1    
neighbors+=1
k=len(neighbors.columns)-1


recommendation_dict = predict(ratings,neighbors, k)

with open('data/recommendation.data', 'w') as fp:
    json.dump(recommendation_dict, fp)
