import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ratings = pd.read_csv('data/ratings.csv')
tags = pd.read_csv('data/tags.csv')

movie_freq = ratings['movieId'].groupby(ratings['movieId']).agg('count').sort_values(ascending=False).reset_index(drop=True)
movie_freq.plot()
plt.ylabel('Viewing Frequency')
plt.xlabel('Movies')
plt.show()
plt.gcf().clear()

user_freq = ratings['userId'].groupby(ratings['userId']).agg('count').sort_values(ascending=False).reset_index(drop=True)
user_freq.plot()
plt.ylabel('Viewing Frequency')
plt.xlabel('Users')
plt.show()
plt.gcf().clear()

tag_freq = tags['tag'].value_counts()
tag_freq.plot()
plt.ylabel('Tag Frequency')
plt.show()
plt.gcf().clear()