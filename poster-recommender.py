from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load the precomputed embeddings, movie indices and movie data
embeddings = np.load('data/clip_poster_embeddings.npy')
emb_df = pd.read_csv('data/embeddings_movie_id.csv', index_col=0)
movies = pd.read_csv('data/movieapp_movie.csv').set_index('movie_id')

# Create movie_id - embeddings mapping
emb_df['embeddings'] = list(embeddings)

# Convert the genres column to Genres
movies['genres'] = movies['genres'].fillna('')
movies['genres'] = movies['genres'].apply(lambda x: x.split(','))

# Explode the genres column in movies DataFrame
movies_exploded = movies.explode('genres')

# Merge the movies DataFrame with the embeddings
movies_with_posters = pd.merge(
    movies_exploded,
    emb_df[['movie_id', 'embeddings']],
    left_on='movie_id',  # adjust if your movie ID column is named differently
    right_on='movie_id'
)


def recommend_by_poster(movie_id, top_n=5):
    """ Recommend movies based on the poster of a given movie ID
        ARGS: movie_id (MovieLens), top_n movies
        RETURNS: MovieLens IDs
    """

    target_rows = movies_with_posters[movies_with_posters['movie_id'] == movie_id]
    if target_rows.empty:
        print(f"No movie found with ID {movie_id}.")
        return None
    
    # Get all unique genres for the target movie
    target_genres = set(target_rows['genres'])

    # Use the first embedding
    target_emb = target_rows.iloc[0]['embeddings'].reshape(1, -1)

    # Filter candidates based on genres and exclude the target movie
    candidates = movies_with_posters[
        (movies_with_posters['movie_id'] != movie_id) &
        (movies_with_posters['genres'].isin(target_genres))
    ].copy()

    # Drop the duplicated movies from exploded df
    candidates = candidates.drop_duplicates('movie_id')


    if candidates.empty:
        print("No other movies with matching genres.")
        return None
    
    # Compute cosine similarity
    embds = np.stack(candidates['embeddings'].values)
    sims = cosine_similarity(target_emb, embds)[0]
    
    candidates['similarity'] = sims

    recommendations = candidates.sort_values('similarity', ascending=False).head(top_n)
    return recommendations['movie_id'].tolist()


movie_id = 1
print(f"Selected movie title: {movies.loc[movie_id]['title']}")
recommendations = recommend_by_poster(movie_id=1)
print(movies.loc[recommendations]['title'])
