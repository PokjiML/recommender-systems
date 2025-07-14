import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the user ratings data

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")


def collaborative_recommend(movie_id, ratings_df, movies_df, top_n=5, min_co_count=50):
    """ Recommend movies using co-rating frequency """
    
    # 1. Find users who rated the given movie
    users_who_rated = ratings_df[ratings_df['movieId'] == movie_id]['userId'].unique()
    
    # 2. Find all other movies these users have rated
    co_rated = ratings_df[(ratings_df['userId'].isin(users_who_rated)) & 
                         (ratings_df['movieId'] != movie_id)]
    
    # 3. Count how often each movie is co-rated
    co_counts = co_rated.groupby('movieId').size().reset_index(name='co_count')
    
    # 4. Add total rating counts for each movie
    movie_rating_counts = ratings_df.groupby('movieId').size().reset_index(name='total_count')
    co_counts = co_counts.merge(movie_rating_counts, on='movieId')
    
    # 5. Normalize co_count by total_count
    co_counts['co_count_norm'] = co_counts['co_count'] / co_counts['total_count']
    
    # 6. Merge with movie titles
    co_counts = co_counts.merge(movies_df, on='movieId')
    
    # 7. Filter to only movies with at least min_co_count co-ratings
    filtered_co_counts = co_counts[co_counts['co_count'] >= min_co_count]
    
    # 8. Recommend top N most co-rated movies (by normalized co-rating)
    recommendations = filtered_co_counts.sort_values('co_count_norm', ascending=False).head(top_n)
    
    return recommendations


# Example usage:
movie_id = 1
print(f"Selected movie title: {movies[movies['movieId'] == movie_id]['title'].values[0]}")
recommendations = collaborative_recommend(movie_id, ratings, movies, top_n=5, min_co_count=50)
print(recommendations.title)