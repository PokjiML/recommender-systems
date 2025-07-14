# Movie Recommender Evaluation

The main scope of this project is to test different recommending strategies against each other.

---

## 📊 Data

Due to file size limits on GitHub, the data is not included in this repository.  
The datasets used by the notebooks can be downloaded from:  
👉 [GroupLens MovieLens datasets](https://grouplens.org/datasets/movielens/)

---

## 🧠 Recommenders

### ✅ Content-based
* **What:** Compute cosine similarity on pre-computed item features (genres, crew, etc.).
* **File:** `content_based_recommender.py`
* **How:**  
  * Loads `movieapp_movie.csv` (via pandas) and `all_features.npy` (via NumPy)
  * Computes cosine similarity (sklearn)
  * Normalizes combined scores with `MinMaxScaler`
  * Returns top-5 recommendations

---

### 📝 Plot-based
* **What:** Compute cosine similarity on TF–IDF–style plot embeddings.
* **File:** `merged_recommender.py`
* **How:**  
  * Loads `movie_plot_embeddings.csv` into DataFrame
  * Extracts embedding for given movie ID
  * Computes cosine similarity (sklearn) against all vectors
  * Reindexes to align with movie list

---

### 🖼 Poster-based
* **What:** Compute cosine similarity on CLIP poster embeddings filtered by genre.
* **File:** `poster_recommender.py`
* **How:**  
  * Loads `clip_poster_embeddings.npy` (NumPy) and `embeddings_movie_id.csv` (pandas)
  * Maps IDs → vectors
  * Filters by shared genres
  * Computes cosine similarity (sklearn)
  * Picks top-5 recommendations

---

### 👥 Collaborative filtering
* **What:** Recommend by normalized co-rating frequency among users.
* **File:** `collaborate_recommender.py`
* **How:**  
  * Loads `ratings.csv` and `movies.csv` via pandas
  * Finds users who rated the target
  * Counts co-rated movies and merges with total rating counts
  * Computes normalized co-count (`co_count / total_count`)
  * Filters by minimum co-count, sorts by score
  * Returns top-5 recommendations

---

### 🔀 Hybrid
* **What:** Weighted sum of Content (0.4), Plot (0.3) and Poster (0.3) similarities.
* **File:** `merged_recommender.py`
* **How:**  
  * Loads `movieapp_movie.csv`, `all_features.npy`, `movie_plot_embeddings.csv`, `embeddings_movie_id.csv`
  * Computes three cosine similarity arrays
  * Normalizes each with `MinMaxScaler`
  * Combines with tunable weights `(0.4, 0.3, 0.3)`
  * Drops the query movie itself
  * Returns top-N recommendations

---

## 📈 Findings

Recommendations were evaluated across five methods using movie examples:

| Recommender   | Excellent | Good | Poor |
|---------------|:--------:|:----:|:----:|
| Plot          | 27 (21.6%) | 40 (32.0%) | 58 (46.4%) |
| Content       | 11 (8.8%)  | 52 (41.6%) | 62 (49.6%) |
| Poster        | 11 (8.8%)  | 63 (50.4%) | 51 (40.8%) |
| Collaborate   | 0 (0.0%)   | 38 (30.4%) | 87 (69.6%) |
| Hybrid        | 12 (9.6%)  | 55 (44.0%) | 58 (46.4%) |

---

