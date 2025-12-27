import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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

# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize numeric features to zero mean and unit variance.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[NUMERIC_COLUMNS])

    return pd.DataFrame(
        scaled,
        columns=NUMERIC_COLUMNS,
        index=df.index
    )


def build_feature_matrix(
    df: pd.DataFrame,
    genre_weight: float = 2.0,
    numeric_weight: float = 0.5
) -> pd.DataFrame:
    """
    Combine weighted genre and numeric features into a single matrix.
    """
    numeric_scaled = scale_numeric_features(df)

    genre_features = df[GENRE_COLUMNS] * genre_weight
    numeric_features = numeric_scaled * numeric_weight

    features = pd.concat([genre_features, numeric_features], axis=1)
    features.index = df.id

    return features

# -------------------------------------------------
# Pivot Table & Normalization
# -------------------------------------------------

def create_movie_user_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-centered movie-user matrix.
    """
    pt = ratings.pivot_table(
        index='id',
        columns='userId',
        values='rating',
        aggfunc='mean'
    )

    # Remove user bias
    pt = pt.sub(pt.mean(axis=0), axis=1)
    pt.fillna(0, inplace=True)

    return pt

# -------------------------------------------------
# Find Common Movies
# -------------------------------------------------

def find_common(
        movie_user: pd.DataFrame,
        features: pd.DataFrame
):
    common_ids = movie_user.index.intersection(features.index)
    movie_user = movie_user.loc[common_ids]
    features = features.loc[common_ids]

    return movie_user, features

# -------------------------------------------------
# User profile construction
# -------------------------------------------------

def build_user_profile(
    item_matrix: pd.DataFrame,
    user_ratings: pd.Series
) -> np.ndarray:
    """
    Build a user profile as a weighted average of item vectors.

    item_matrix: rows = items (movies), columns = features/users
    user_ratings: ratings given by the user (indexed by item id)
    """
    rated_items = user_ratings[user_ratings != 0]

    if rated_items.empty:
        raise ValueError("User has no ratings")

    weights = rated_items.values.reshape(-1, 1)
    item_vectors = item_matrix.loc[rated_items.index].values

    profile = np.sum(item_vectors * weights, axis=0)
    profile /= np.sum(np.abs(weights))

    return profile.reshape(1, -1)


# -------------------------------------------------
# Similarity scoring
# -------------------------------------------------

def compute_similarity_scores(
    user_profile: np.ndarray,
    item_matrix: pd.DataFrame,
    exclude_items: pd.Index
) -> pd.Series:
    """
    Compute cosine similarity between a user profile and all items.
    """
    similarities = cosine_similarity(user_profile, item_matrix.values)[0]

    scores = pd.Series(similarities, index=item_matrix.index)
    scores = scores.drop(exclude_items, errors="ignore")

    return scores


# -------------------------------------------------
# Hybrid user-based recommendation
# -------------------------------------------------

def recommend_for_user(
    user_id: int,
    ratings_matrix: pd.DataFrame,
    content_features: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10,
    alpha: float = 0.7
) -> list[str]:
    """
    Recommend movies for a user using user-based hybrid filtering.

    alpha = weight for collaborative filtering
    (1 - alpha) = weight for content-based filtering
    """
    if user_id not in ratings_matrix.columns:
        raise ValueError("User not found")

    user_ratings = ratings_matrix[user_id]

    rated_movie_ids = user_ratings[user_ratings != 0].index

    # --- Collaborative user profile ---
    cf_profile = build_user_profile(
        item_matrix=ratings_matrix,
        user_ratings=user_ratings
    )

    cf_scores = compute_similarity_scores(
        user_profile=cf_profile,
        item_matrix=ratings_matrix,
        exclude_items=rated_movie_ids
    )

    # --- Content-based user profile ---
    cb_profile = build_user_profile(
        item_matrix=content_features,
        user_ratings=user_ratings
    )

    cb_scores = compute_similarity_scores(
        user_profile=cb_profile,
        item_matrix=content_features,
        exclude_items=rated_movie_ids
    )

    # --- Hybrid combination ---
    hybrid_scores = alpha * cf_scores + (1 - alpha) * cb_scores

    top_ids = hybrid_scores.sort_values(ascending=False).head(n).index

    # Preserve ranking order
    return (
        movies_df
        .set_index("id")
        .loc[top_ids]["original_title"]
        .tolist()
    )

# -------------------------------------------------
# Pipeline Entry Point
# -------------------------------------------------

def build_recommender():
    movies_data = load_content_data("Recommendation_data.csv")

    ratings = load_ratings_data("ratings.csv", "links.csv")
    ratings = filter_ratings(ratings)
    movie_user_matrix = create_movie_user_matrix(ratings)

    features = build_feature_matrix(movies_data)

    movie_user_matrix, features = find_common(movie_user_matrix, features)

    return movie_user_matrix, features, movies_data
# -------------------------------------------------
# Example Usage
# -------------------------------------------------

if __name__ == "__main__":

    movie_user_matrix, features, movies_data = build_recommender()

    try:
        user_id = int(input("Enter user ID: ").strip())
        if user_id not in movie_user_matrix.columns:
            raise ValueError("User ID not found in dataset")

        recommendations = recommend_for_user(
            user_id=user_id,
            ratings_matrix=movie_user_matrix,
            content_features=features,
            movies_df=movies_data,
            n=10,
            alpha=0.7
        )

        print("\nRecommended movies:")
        for i, title in enumerate(recommendations, 1):
            print(f"{i}. {title}")

    except ValueError as e:
        print(f"Error: {e}")
