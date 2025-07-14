import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ---- Download all necessary files -----
 
# Movie Dataset
movies = pd.read_csv('data/movieapp_movie.csv')
movies.set_index('movie_id', inplace=True)

# Metadata features
content_based_features = np.load('data/all_features.npy')

# Plot embeddings
plot_embeddings = pd.read_csv('data/movie_plot_embeddings.csv')
plot_embeddings.set_index('movieId', inplace=True)

# Poster embeddings
clip_embeddings_df = pd.read_csv('data/embeddings_movie_id.csv')
clip_embeddings = np.load('data/clip_poster_embeddings.npy')
clip_embeddings_df['embeddings'] = list(clip_embeddings)
clip_embeddings_df.set_index('movie_id', inplace=True)

# ---- Movie Recommender ----

def merged_recommendations(movie_id, top_n=5, weights=(0.4, 0.3, 0.3)):
    """
    Combine content, plot, and poster similarities for recommendations.
    weights: (content_weight, plot_weight, poster_weight)
    """
    # --- Content-based ---
    try:
        movie_idx = movies.index.get_loc(movie_id)
    except KeyError:
        print(f"Movie ID {movie_id} not found in movies.")
        return []
    target_vector = content_based_features[movie_idx].reshape(1, -1)
    content_sims = cosine_similarity(target_vector, content_based_features).flatten()
    
    # --- Plot-based ---
    if movie_id not in plot_embeddings.index:
        print(f"Movie ID {movie_id} not found in plot embeddings.")
        return []
    plot_target = plot_embeddings.loc[movie_id].values.reshape(1, -1)
    plot_sims = cosine_similarity(plot_target, plot_embeddings)[0]
    
    # --- Poster-based ---
    poster_row = clip_embeddings_df[clip_embeddings_df.index == movie_id]
    if poster_row.empty:
        print(f"Movie ID {movie_id} not found in poster embeddings.")
        return []
    
    poster_emb = poster_row.iloc[0]['embeddings'].reshape(1, -1)
    poster_sims = cosine_similarity(poster_emb, clip_embeddings).flatten()
    
    # --- Align indices ---
    all_movie_ids = list(movies.index)
    # Ensure all arrays are aligned to movies DataFrame index
    content_sims = pd.Series(content_sims, index=all_movie_ids)
    plot_sims = pd.Series(plot_sims, index=plot_embeddings.index)
    poster_sims = pd.Series(poster_sims, index=clip_embeddings_df.index)
    
    # Reindex plot and poster to match movies
    plot_sims = plot_sims.reindex(all_movie_ids, fill_value=0)
    poster_sims = poster_sims.reindex(all_movie_ids, fill_value=0)
    
    # --- Normalize ---
    scaler = MinMaxScaler()
    content_norm = scaler.fit_transform(content_sims.values.reshape(-1, 1)).flatten()
    plot_norm = scaler.fit_transform(plot_sims.values.reshape(-1, 1)).flatten()
    poster_norm = scaler.fit_transform(poster_sims.values.reshape(-1, 1)).flatten()
    
    # --- Weighted sum ---
    combined_score = (
        weights[0] * content_norm +
        weights[1] * plot_norm +
        weights[2] * poster_norm
    )
    
    # Exclude the target movie itself
    result_df = pd.DataFrame({
        'movie_id': all_movie_ids,
        'score': combined_score
    }).set_index('movie_id')
    result_df = result_df.drop(movie_id)
    
    # Top N recommendations
    top_recs = result_df.sort_values('score', ascending=False).head(top_n)
    return movies.loc[top_recs.index].index.tolist()


# Example usage
movie_id = 1
print(f"Selected movie title: {movies.loc[movie_id]['title']}")
recommendations = merged_recommendations(movie_id, top_n=5)
print(movies.loc[recommendations]['title'])