#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Q1

#IMPORT THE NECESSARY LIBRARIES
#IMPORT THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
import json as js
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

# DATA DESCRIPTION 

# OPEN THE CSV FILES AND CONVERT THEM TO DATAFRAMES

##links=pd.read_csv('links.csv', sep=',',encoding='latin-1')
##movies=pd.read_csv('movies.csv', sep=',',encoding='latin-1')
ratings=pd.read_csv('data/ratings.csv', sep=',',encoding='latin-1')
##tags=pd.read_csv('tags.csv', sep=',',encoding='latin-1')

# Q2

# FIND SIMILARITIES - 1

# IMPORT RATINGS CSV AGAIN IN ORDER TO MAKE A NEW DATAFRAME WHICH IS GOING TO BE USED FOR RECOMMENDATION
ratings_sim=pd.read_csv('data/ratings.csv', sep=',',encoding='latin-1')
# REMOVE WHITESPACE SPACE FROM COLUMN NAMES 
ratings_sim.col = ratings_sim.columns.str.strip()
# NEW DATAFRAME --> COLUMNS(MOVIE_IDS) AND ROWS (USER_IDS)/RATINGS
ratings_new=ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print ('The ratings matrix presents movies Ids as colums and user Ids as rows and has dimensions ' + str(ratings_new.shape)) 
# STORE THE RATINGS_NEW IN A CSV FILE
ratings_new.to_csv('data/pairs-100k-movielens.data', sep='\t', encoding='utf-8')
# STORE THE RATING DATAFRAME AS ARRAY
ratings_mat = ratings_new.as_matrix()
##np.savetxt("pairs-100k-movielens.data", R, delimiter=",")
user_ratings_mean = np.mean(ratings_mat, axis = 1)
ratings_matr_demeaned = ratings_mat - user_ratings_mean.reshape(-1, 1)

# FIND SIMILARITIES - 2

def split(ratings):
    test=np.zeros(ratings.shape)
    cpy=ratings.copy()
    for user1 in range(ratings.shape[0]):
        test_list = np.random.choice(ratings[user1, :].nonzero()[0], size=10, replace=False)
        cpy[user1,test_list]=0
        test[user1,test_list]=ratings[user1,test_list]              
    return cpy,test
cpy_list, test_list = split(ratings_mat)
print ('The copying set has dimensions ' + str(cpy_list.shape))
print ('The test set has dimensions ' + str(test_list.shape))

k=5
neighbour = NearestNeighbors(k,'cosine')      
neighbour.fit(cpy_list)
maxk_dist,maxk_user1 = neighbour.kneighbors(cpy_list, return_distance=True)
print (maxk_user1.shape)

colum = ['UserId','Neighbour1','Neighbour2','Neighbour3','Neighbour4']
maxk_user1=pd.DataFrame(maxk_user1, columns=colum)

# MAKE A DICTIONARY OF THE ABOVE
maxk_user1_dict= maxk_user1.to_dict(orient='index')
# STORE THE ABOVE AS JSON FILE
import json
with open('data/neighbors-k-100-movilens.data', 'w') as fp:
    json.dump(maxk_user1_dict, fp)

