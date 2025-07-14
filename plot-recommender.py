import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



# Get dataframe with movie features
movie_content_df = pd.read_csv('/home/movie_content_df.csv')
movie_content_df.set_index('movieId', inplace=True)

# Data frame with movie plot embeddings
plot_embeddings = pd.read_csv('/home/movie_plot_embeddings.csv')
plot_embeddings.set_index('movieId', inplace=True)

# Data frame connecting the movieIDs and Names
movie_names = movie_content_df['title']

# List of all movie Ids
movie_ids = plot_embeddings.index.tolist()



def get_similar_movies(movie_id, n=5):
    """ Return the titles of N most similar movies """

    # If the movie can't be found
    if movie_id not in movie_ids:
        return f"Movie ID {movie_id} not found in dataset"
    
    # get the selected movie embeddings
    target_embedding = plot_embeddings.loc[movie_id].values.reshape(1, -1)
    
    # Compute the similarite between chosen movie and all other
    similarities = cosine_similarity(target_embedding, plot_embeddings)[0]

    # A list of (movie_id, similarity)
    similarity_pairs = list(zip(plot_embeddings.index, similarities))

    # Sort the movies by similarity
    similarity_pairs.sort(key=lambda x: x[1], reverse=True)

    # Choose top N movies (without the first one = the same movie)
    top_similar = similarity_pairs[1:n+1]
    top_movie_ids = [movie_id for movie_id, _ in top_similar]

    # The ids and names of recommended movies
    top_movie_names = movie_names[top_movie_ids]

    print(f"Top {n} most similar movies for {movie_names[movie_id]}")

    return top_movie_names


recommendations = get_similar_movies('2')
print(recommendations)