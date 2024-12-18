# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:07:56 2024

@author: marty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras import ops
import streamlit as st

"""
## First, load the data and apply preprocessing
"""

df = pd.read_csv("FR_youtube_trending_data.csv")
df = df.drop_duplicates(subset='video_id')
# Create a new column 'url' that contains the YouTube URL for each video_id
df['url'] = 'https://www.youtube.com/watch?v=' + df['video_id'].astype(str)

# Define the number of users you want to create
num_users = 1000  # Set the number of unique users you'd like to generate

# Generate user_ids
user_ids = [f"user_{i+1}" for i in range(num_users)]

# Créer une liste pour stocker les données
data = []

# For each user, select 5 random videos
for user_id in user_ids:
    selected_videos = df.sample(n=50, replace=False) #pas de répétition de vidéos
    
    # Add it in the list
    for index, row in selected_videos.iterrows():
        data.append({
            'user_id': user_id,
            'video_id': row['video_id'],
            'title': row['title'],
            'rating': round(np.random.uniform(0.5, 5), 1)  # Ajouter un ranking aléatoire, arrondi à 1 décimale
        })

# Convert the list in DataFrame
user_video_df = pd.DataFrame(data)

# Save to a new CSV file
user_video_df.to_csv('user_video_mapping.csv', index=False)

"""
First, need to perform some preprocessing to encode users and videos as integer indices.
"""
user_ids = user_video_df["user_id"].unique().tolist() #covertit colonne user_id en une liste pour avoir une reference pour chaque utilisateur unique
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

yt_ids = user_video_df["video_id"].unique().tolist() #même chose pour les vidéos
video2video_encoded = {x: i for i, x in enumerate(yt_ids)}
video_encoded2video = {i: x for i, x in enumerate(yt_ids)}

user_video_df["user"] = user_video_df["user_id"].map(user2user_encoded)
user_video_df["video"] = user_video_df["video_id"].map(video2video_encoded)

num_users = len(user2user_encoded)
num_videos = len(video_encoded2video)

user_video_df["rating"] = user_video_df["rating"].values.astype(np.float32) #float 32 améliore la memoire et la compatibilité avec modèle

df.replace(0,1,inplace = True)

# Step 1: Calculate the likes ratio
df['like_over_view_ratio'] = df['likes'] / df['view_count']  # Add small value to avoid division by zero

# Step 2: Define the threshold
threshold = 0.08  # Example threshold value

# Step 3: Apply the adjustment to user_video_df
def adjust_ratings(row):
    # If the video's likes ratio is above the threshold and the rating is below 3, set it to 3
    video_likes_ratio = df.loc[df['video_id'] == row['video_id'], 'like_over_view_ratio'].values[0]
    if video_likes_ratio > threshold and row['rating'] < 3:
        return 3
    return row['rating']

# Apply the function to update ratings
user_video_df['rating'] = user_video_df.apply(adjust_ratings, axis=1)

# Verify the changes
print(user_video_df.head())

# min and max ratings will be used to normalize the ratings later
min_rating = min(user_video_df["rating"])
max_rating = max(user_video_df["rating"])

print(
    "Number of users: {}, Number of Videos: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_videos, min_rating, max_rating
    )
)

"""
## Prepare training and validation data
"""
user_video_df = user_video_df.sample(frac=1, random_state=42)
x = user_video_df[["user", "video"]].values #x=tableau avec les pairs de users et de videos

# Normalize the targets between 0 and 1. Makes it easy to train.
y = user_video_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values 

# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.8 * user_video_df.shape[0])

x_train, x_val, y_train, y_val = (
    x[:train_indices], #données d'entrée (entrainement)
    x[train_indices:], #données d'entrée (validation)
    y[:train_indices], #données cibles (entrainement)
    y[train_indices:], #données cibles (validation)
)

"""
## Create the model

We embed both users and videos in to 50-dimensional vectors.

The model computes a match score between user and video embeddings via a dot product,
and adds a per-video and per-user bias. The match score is scaled to the [0, 1]
interval via a sigmoid (since our ratings are normalized to this range).
"""
embedding_size = 30  #user and videos represented by vectors with a 50 dimension


class RecommenderNet(keras.Model): #creation of a personnalized model
    def __init__(self, num_users, num_videos, embedding_size, **kwargs):
        super().__init__(**kwargs) #super() calls the class of keras.model
        self.num_users = num_users #makes the layer accessible embedding size
        self.num_videos = num_videos #makes the layer accessible embedding size
        self.embedding_size = embedding_size #makes the layer accessible for the embedding size
        
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="glorot_normal", #randomize every weights for the initialization
            embeddings_regularizer=keras.regularizers.l2(1e-3), #undermine overfitting
        )
        self.user_bias = layers.Embedding(num_users, 1) #improve precision for the recommandation by adjusting the score for each user
        
        self.videos_embedding = layers.Embedding(
            num_videos,
            embedding_size,
            embeddings_initializer="glorot_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-3),
        )
        self.video_bias = layers.Embedding(num_videos, 1)

    def call(self, inputs):
        #recuperation of embeddings vectors for a user and a specific video
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        video_vector = self.videos_embedding(inputs[:, 1])
        video_bias = self.video_bias(inputs[:, 1])
        
        dot_user_video = ops.tensordot(user_vector, video_vector, 2) #scalar product between vector user and video for a similarity test
        
        # Add all the components (including bias)
        x = dot_user_video + user_bias + video_bias
        
        # activation sigmoid limit on the final score between 0 and 1
        return ops.nn.sigmoid(x)


model = RecommenderNet(num_users, num_videos, embedding_size)
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
)

"""
## Train the model based on the data split
"""
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64   ,
    epochs=10,
    verbose=1,
    validation_data=(x_val, y_val),
)

"""
## Plot training and validation loss
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss") 
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

"""
## Show top 10 video recommendations to a user
"""
# Let us get a user and see the top recommendations.
rand_user = user_video_df.user_id.sample(1).iloc[0]
videos_watched_by_user = user_video_df[user_video_df.user_id == rand_user]
videos_not_watched = user_video_df[
    ~user_video_df["video_id"].isin(videos_watched_by_user.video_id.values)
]["video_id"].unique()

# Convert the video list to drop duplicates
videos_not_watched = list(set(videos_not_watched).intersection(set(video2video_encoded.keys())))
videos_not_watched = [[video2video_encoded.get(x)] for x in videos_not_watched]

user_encoder = user2user_encoded.get(rand_user)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(videos_not_watched), videos_not_watched)
)

# Predict the scores for the unwatched videos
ratings = model.predict(user_movie_array).flatten()

# Get the 10 best ratings
top_ratings_indices = np.unique(ratings.argsort()[-10:][::-1])  # Utiliser np.unique pour garantir l'unicité

# Convert it into ID's
recommended_video_ids = [
    video_encoded2video.get(videos_not_watched[x][0]) for x in top_ratings_indices
]

print("Showing recommendations for user: {}".format(rand_user))
print("====" * 9)
print("Videos with high ratings from user")
print("----" * 8)
top_videos_user = (
    videos_watched_by_user.sort_values(by="rating", ascending=False)
    .drop_duplicates(subset=["video_id"])  # Ensure unique video IDs
    .head(50)
    .video_id.values
)
video_df_rows = user_video_df[user_video_df["video_id"].isin(top_videos_user)]
for row in video_df_rows.itertuples():
    print(row.title)

print("----" * 8)
print("Top 10 video recommendations")
print("----" * 8)
recommended_videos = user_video_df[user_video_df["video_id"].isin(recommended_video_ids)].drop_duplicates(subset=["video_id"]) 
for row in recommended_videos.drop_duplicates("video_id").itertuples():  # Utiliser drop_duplicates pour éviter les doublons
    print(row.title)
    

# Get the rows from the original DataFrame where the 'video_id' is in the recommended list and remove duplicates
recommended_videos_df = df[df['video_id'].isin(recommended_video_ids)]

# Extract the URLs of the recommended videos
recommended_video_urls = recommended_videos_df['url'].tolist()

# Print the URLs
print("Top 10 recommended video URLs:")
for url in recommended_video_urls:
    print(url)

video_url = recommended_video_urls[0]

# Add streamlit to show the results
def main():
    st.title("Système de Recommandation Vidéo 🎥")

    # Choose a random user
    st.sidebar.header("Options utilisateur")
    rand_user = st.sidebar.selectbox(
        "Sélectionnez un utilisateur",
        user_video_df["user_id"].unique().tolist()
    )

    # Show the videos already watched by the user
    st.subheader(f"Vidéos regardées par l'utilisateur {rand_user}")
    videos_watched = user_video_df[user_video_df.user_id == rand_user]
    for row in videos_watched.itertuples():
        st.markdown(f"**{row.title}**")
        st.markdown(f"[Lien vers la vidéo](<{df[df['video_id'] == row.video_id]['url'].iloc[0]}>)", unsafe_allow_html=True)

    # Generate the recommandations
    st.subheader(f"Top 10 vidéos recommandées pour l'utilisateur {rand_user}")
    recommended_videos_df = df[df['video_id'].isin(recommended_video_ids)]
    for row in recommended_videos_df.itertuples():
        st.markdown(f"**{row.title}**")
        st.video(row.url)
        st.markdown(f"[Lien vers la vidéo](<{row.url}>)", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
