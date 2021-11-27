#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans


songs = pd.read_csv('all_tracks_final.csv')
top100 = pd.read_csv('Top100_songs.csv')
df_all_features = pd.read_csv('df_all_features.csv')

secrets_file = open("secrets.txt","r")
string = secrets_file.read()
string.split('\n')
secrets_dict={}
for line in string.split('\n'):
    if len(line) > 0:
        secrets_dict[line.split(':')[0]]=line.split(':')[1]
secrets_dict
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=secrets_dict['cid'],
                                                           client_secret=secrets_dict['csecret']))

x_transformer = StandardScaler().fit(df_all_features)
x_prep = x_transformer.transform(df_all_features)
kmeans = KMeans(n_clusters=3, random_state=1234)
kmeans.fit(x_prep)
clusters = kmeans.predict(x_prep)
pd.Series(clusters).value_counts().sort_index()

song_choice = input('Please enter a song name: ')
if top100.song.isin([song_choice]).any:
    yoursong = top100.sample(n=1)
    print('We recommend: ' + yoursong['song'].values[0] + ' by ' +  yoursong['artist'].values[0])
else:
    
    results = sp.search(song_choice, type = 'track', limit=1)
    song_uri =song_uri = results['tracks']['items'][0]['uri']
    features = sp.audio_features(song_uri) # this should be a dictionary
    features_df = pd.DataFrame(features)
    features_df = features_df.select_dtypes(np.number)
    features_df = features_df.drop(['duration_ms', 'time_signature'], axis=1)
    x_new = x_transformer.transform(features_df)
    cluster_new = kmeans.predict(x_new)# this is an array
    df_cluster = all_tracks_final['cluster'] == list(cluster_new)[0]
    print('We recommend: ' + df_cluster.sample(n=1))


# In[ ]:




