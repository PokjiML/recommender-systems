import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the precomputed features and movie data
movies = pd.read_csv('data/movieapp_movie.csv').set_index('movie_id')
all_features = np.load('data/all_features.npy')


def content_based_recommender(movie_id):
    """ Return a list of recommended movie IDs based on content
        similarity and their avarage rating.
    """

    if movie_id not in movies.index:
        print(f"No movie found with ID {movie_id}.")
        return None
    
    movie_idx = movies.index.get_loc(movie_id)

    target_vector = all_features[movie_idx].reshape(1, -1)
    similarities = cosine_similarity(target_vector, all_features).flatten()

    # Get the 10 most similar movies
    most_similar_indices = similarities.argsort()[::-1][1: 11] 

    # Map indices to movieIds
    movie_ids = movies.index.values
    most_similar_movie_ids = movie_ids[most_similar_indices]

    # Take into account the rating of the movie
    ratings = movies.loc[most_similar_movie_ids]['avgRating']
    ratings_scaled = MinMaxScaler().fit_transform(ratings.values.reshape(-1, 1)).flatten()

    # Combine similarity and rating
    sim_weight = 0.6
    rating_weight = 0.4
    combined_score = (similarities[most_similar_indices] * sim_weight) + (ratings_scaled * rating_weight)


    sorted_idx = np.argsort(combined_score)[::-1][:5]
    recommended_movie_ids = most_similar_movie_ids[sorted_idx]


    return recommended_movie_ids


movie_id = content_based_recommender(1)
print(f"For movie title: {movies.loc[movie_id].title}")

movie_names = movies.loc[movie_id]['title'].values
print(movie_names)

