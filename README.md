# Hybrid Movie Recommendation System

This project implements **hybrid movie recommendation systems** that combine **content-based filtering (CB)** and **collaborative filtering (CF)** using cosine similarity.

Two approaches are provided:

- **Item-based hybrid recommendation**

- **User-based hybrid recommendation**


## Files
```bash
.
├── recommendation_CF+CB.py             # Item-based hybrid recommender
├── recommendation_hybrid_user_based.py # User-based hybrid recommender
├── Recommendation_data.csv             # Movie features (content-based)
├── ratings.csv                         # MovieLens ratings
├── links.csv                           # Mapping MovieLens → TMDB IDs
└── README.md
```

## Recommendation Approaches
**1. Item-Based Hybrid (CF + CB)**

 **File:** `recommendation_CF+CB.py`

 - Builds a content similarity matrix using:

   - Movie genres

   - Numeric features (runtime, popularity, votes, etc.)

- Builds an item-based collaborative similarity matrix from user ratings

- Combines both similarities using a weighted hybrid score

- Recommends movies similar to a given movie title

**Input:** movie title

**Output:** list of similar movies

**2. User-Based Hybrid (CF + CB)**

**File:** `recommendation_hybrid_user_based.py`

- Constructs user profiles from:

   - Collaborative filtering (user–movie ratings)

   - Content-based features

- Computes similarity between user profiles and movies

- Produces personalized recommendations for a given user

**Input:** user ID

**Output:** list of recommended movies

## Data Requirements

- `Recommendation_data.csv`

   Movie metadata with genre indicators and numeric features

- `ratings.csv`
  
   MovieLens user ratings

- `links.csv`

   Mapping between MovieLens IDs and TMDB IDs

# How to Run
**Item-Based Recommendation**

```bash
python recommendation_CF+CB.py
```

**User-Based Recommendation**

```bash
python recommendation_hybrid_user_based.py
```

