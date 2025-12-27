import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# -------------------------------------------------
# Configuration
# -------------------------------------------------

GENRE_COLUMNS = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History',
    'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Science Fiction',
    'TV Movie', 'Thriller', 'War', 'Western'
]

NUMERIC_COLUMNS = [
    'popularity', 'runtime', 'vote_average', 'vote_count',
    'log_budget', 'movie_age', 'release_quarter',
    'pop_actor', 'pop_director'
]

# -------------------------------------------------
# Data Loading
# -------------------------------------------------

def load_content_data(path: str) -> pd.DataFrame:
    """
    Load content-based movie features.
    """
    df = pd.read_csv(path)
    df['id'] = df['id'].astype(int)
    return df


def load_ratings_data(ratings_path: str, links_path: str) -> pd.DataFrame:
    """
    Load ratings and map MovieLens IDs to TMDB IDs.
    """
    ratings = pd.read_csv(ratings_path).drop(columns=['timestamp'])
    links = pd.read_csv(links_path).drop(columns=['imdbId'])
    links.rename(columns={'tmdbId': 'id'}, inplace=True)

    ratings = ratings.merge(
        links[['movieId', 'id']],
        on='movieId',
        how='inner'
    )
    # Drop rows without a valid TMDB id
    ratings = ratings.dropna(subset=['id'])
    ratings['id'] = ratings['id'].astype(int)
    return ratings

# -------------------------------------------------
# Content-Based Similarity
# -------------------------------------------------

def build_content_similarity(df: pd.DataFrame,
                             genre_weight: float = 2.0,
                             numeric_weight: float = 0.5) -> pd.DataFrame:
    """
    Build cosine similarity matrix using content features.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[NUMERIC_COLUMNS])
    numeric_scaled = pd.DataFrame(scaled, columns=NUMERIC_COLUMNS, index=df.index)

    genre_features = df[GENRE_COLUMNS] * genre_weight
    numeric_features = numeric_scaled * numeric_weight

    feature_matrix = pd.concat([genre_features, numeric_features], axis=1)
    sim = cosine_similarity(feature_matrix.values)

    return pd.DataFrame(sim, index=df['id'], columns=df['id'])

# -------------------------------------------------
# Collaborative Filtering (Item-Based)
# -------------------------------------------------

def filter_ratings(ratings: pd.DataFrame,
                   min_user_ratings: int = 200,
                   min_movie_ratings: int = 50) -> pd.DataFrame:
    """
    Remove inactive users and unpopular movies.
    """
    active_users = ratings['userId'].value_counts()
    ratings = ratings[ratings['userId'].isin(active_users[active_users > min_user_ratings].index)]

    popular_movies = ratings.groupby('movieId')['rating'].count()
    ratings = ratings[ratings['movieId'].isin(popular_movies[popular_movies > min_movie_ratings].index)]

    return ratings.reset_index(drop=True)


def build_cf_similarity(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Build item-based collaborative filtering similarity matrix.
    """
    pt = ratings.pivot_table(
        index='id',
        columns='userId',
        values='rating',
        aggfunc='mean'
    )

    # Remove user bias
    pt = pt.sub(pt.mean(axis=0), axis=1).fillna(0)

    sparse_matrix = csr_matrix(pt.values)
    sim = cosine_similarity(sparse_matrix)

    return pd.DataFrame(sim, index=pt.index, columns=pt.index)

# -------------------------------------------------
# Hybrid Similarity
# -------------------------------------------------

def build_hybrid_similarity(cb_sim: pd.DataFrame,
                            cf_sim: pd.DataFrame,
                            alpha: float = 0.7) -> pd.DataFrame:
    """
    Combine content-based and collaborative similarities.
    """
    common_ids = cb_sim.index.intersection(cf_sim.index)

    cb_sim = cb_sim.loc[common_ids, common_ids]
    cf_sim = cf_sim.loc[common_ids]

    return alpha * cf_sim + (1 - alpha) * cb_sim

def normalize_title(title: str) -> str:
    """
    Normalize movie titles for reliable matching.
    """
    return title.strip().lower()

# -------------------------------------------------
# Recommendation Logic
# -------------------------------------------------

def recommend_hybrid(
    movie_title: str,
    movies_df: pd.DataFrame,
    hybrid_sim: pd.DataFrame,
    top_n: int = 10
) -> list[str]:
    """
    Recommend movies using hybrid similarity given a movie title.
    """
    movie_title_norm = normalize_title(movie_title)
    matches = movies_df[movies_df['title_norm'] == movie_title_norm]

    if matches.empty:
        raise ValueError(f"Movie '{movie_title}' not found")

    movie_id = int(matches.iloc[0]['id'])

    if movie_id not in hybrid_sim.index:
        raise ValueError("Movie not available in hybrid similarity matrix")

    scores = hybrid_sim.loc[movie_id].drop(movie_id)
    top_ids = scores.sort_values(ascending=False).head(top_n).index

    return movies_df.set_index('id').loc[top_ids]['original_title'].tolist()

# -------------------------------------------------
# Pipeline Entry Point
# -------------------------------------------------

def build_hybrid_recommender():
    movies_df = load_content_data("Recommendation_data.csv")
    movies_df['title_norm'] = movies_df['original_title'].apply(normalize_title)
    ratings = load_ratings_data("ratings.csv", "links.csv")

    ratings = filter_ratings(ratings)

    cb_sim = build_content_similarity(movies_df)
    cf_sim = build_cf_similarity(ratings)

    hybrid_sim = build_hybrid_similarity(cb_sim, cf_sim, alpha=0.7)

    return movies_df, hybrid_sim

# -------------------------------------------------
# Example Usage
# -------------------------------------------------

if __name__ == "__main__":
    movies_df, hybrid_sim = build_hybrid_recommender()

    user_movie = input("Enter a movie name: ").strip()

    try:
        recommendations = recommend_hybrid(
            movie_title=user_movie,
            movies_df=movies_df,
            hybrid_sim=hybrid_sim,
            top_n=10
        )

        print("\nRecommended movies:")
        for movie in recommendations:
            print("-", movie)

    except ValueError as e:
        print(e)

