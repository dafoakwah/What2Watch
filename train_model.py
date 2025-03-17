# scripts/train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import precision_score, recall_score
from collections import Counter
import numpy as np
import ast

def load_engineered_dataset(filepath):
    """
    Load the engineered dataset with optimized data types.
    """
    # Specify data types for problematic columns
    dtype = {
        'genres': 'str',
        'keywords': 'str',
        'overview': 'str',
        'release_year': 'float64'  # Ensure release_year is numeric
    }
    
    # Load the dataset
    df = pd.read_csv(filepath, dtype=dtype, low_memory=False)
    
    # Handle missing or invalid release_year values
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    df = df.dropna(subset=['release_year'])  # Drop rows with missing release_year
    
    return df

def filter_dataset(df, min_votes=50, min_year=1980):
    """
    Filter the dataset to reduce its size.
    """
    print(f"Filtering movies with at least {min_votes} votes and released after {min_year}...")
    
    # Filter movies with at least min_votes votes
    df = df[df['vote_count'] >= min_votes]
    
    # Filter movies released after min_year
    df = df[df['release_year'] >= min_year]
    
    # Reset the index to ensure alignment with the similarity matrix
    df = df.reset_index(drop=True)
    
    print(f"Filtered dataset has {len(df)} movies.")
    return df

def safe_eval(val):
    try:
        if isinstance(val, str):  # Ensure it's a string before evaluation
            return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []  # Return an empty list if parsing fails
    return val  # If already a list, return as is

def create_similarity_matrix(df):
    """
    Create a similarity matrix based on separate processing of text and categorical features.
    """
    # Process textual data separately (overview, tagline, production_companies)
    text_features = df[['overview', 'tagline', 'production_companies']].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Vectorize textual features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_features=5000)  # Remove rare words, limit features
    text_tfidf_matrix = vectorizer.fit_transform(text_features)

    # Apply weights to overview and tagline
    overview_weight = 1.5  # Higher weight for overview
    tagline_weight = 1.2   # Slightly higher weight for tagline
    production_weight = 1.0  # Default weight for production_companies

    # Apply weights to the TF-IDF matrix
    weights = [overview_weight, tagline_weight, production_weight]
    weighted_tfidf_matrix = text_tfidf_matrix.copy()
    for i, weight in enumerate(weights):
        weighted_tfidf_matrix[:, i] = weighted_tfidf_matrix[:, i] * weight

    # Convert genres and keywords into a one-hot encoded matrix
    mlb = MultiLabelBinarizer()

    # Convert stringified lists into real lists
    df['genres'] = df['genres'].apply(safe_eval)  # Convert genres from string to list
    df['keywords'] = df['keywords'].apply(safe_eval)  # Convert keywords from string to list

    genres_matrix = mlb.fit_transform(df['genres'])
    keywords_matrix = mlb.fit_transform(df['keywords'])

    # Combine text embeddings with categorical embeddings
    combined_features_matrix = hstack([weighted_tfidf_matrix, csr_matrix(genres_matrix), csr_matrix(keywords_matrix)])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(combined_features_matrix, combined_features_matrix, dense_output=False)

    return similarity_matrix

def recommend_movies(movie_title, df, similarity_matrix, top_n=10):
    """
    Recommend movies based on similarity to a given movie.
    """
    # Check if the movie exists in the filtered dataset
    if movie_title not in df['title'].values:
        print(f"Movie '{movie_title}' not found in the filtered dataset.")
        return []
    
    # Find the index of the movie in the filtered dataset
    movie_index = df[df['title'] == movie_title].index[0]
    print(f"Index of '{movie_title}' in the filtered dataset: {movie_index}")
    
    # Get similarity scores for the movie
    similarity_scores = list(enumerate(similarity_matrix[movie_index].toarray().flatten()))
    
    # Sort movies by similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N similar movies
    top_movies = similarity_scores[1:top_n + 1]  # Exclude the movie itself
    
    # Return the recommended movies
    recommended_movies = []
    for index, score in top_movies:
        recommended_movies.append({
            'title': df.iloc[index]['title'],
            'genres': df.iloc[index]['genres'],
            'similarity_score': score
        })
    
    return recommended_movies

def evaluate_recommendations(recommendations, ground_truth, k=10):
    """
    Evaluate the recommendations using Precision@K and Recall@K.
    """
    # Convert recommendations and ground truth to binary vectors
    recommended_titles = [movie['title'] for movie in recommendations[:k]]
    y_true = [1 if title in ground_truth else 0 for title in recommended_titles]
    y_pred = [1] * len(recommended_titles)  # All recommendations are considered positive

    # Calculate Precision@K and Recall@K
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return precision, recall

def calculate_diversity_score(recommendations):
    """
    Calculate the diversity score of the recommendations.
    """
    # Extract genres from recommendations
    genres = [genre for movie in recommendations for genre in movie['genres']]
    
    # Calculate the number of unique genres
    unique_genres = set(genres)
    
    # Calculate the diversity score
    diversity_score = len(unique_genres) / len(genres) if genres else 0
    
    return diversity_score

if __name__ == '__main__':
    # Load the engineered dataset
    engineered_filepath = 'data/engineered_movies.csv'
    print("Loading engineered dataset...")
    engineered_df = load_engineered_dataset(engineered_filepath)
    
    # Filter the dataset to reduce its size
    print("Filtering dataset...")
    filtered_df = filter_dataset(engineered_df, min_votes=50, min_year=1980)
    
    # Print the filtered dataset
    print("\nFiltered Dataset:")
    print(filtered_df[['title', 'release_year', 'vote_count']].head())
    
    # Check if 'Inception' is in the filtered dataset
    if 'Inception' in filtered_df['title'].values:
        print("\n'Inception' is in the filtered dataset.")
    else:
        print("\n'Inception' is NOT in the filtered dataset.")
    
    # Create similarity matrix
    print("\nCreating similarity matrix...")
    similarity_matrix = create_similarity_matrix(filtered_df)
    
    # Test the recommendation system
    movie_title = 'Inception'  # Example movie
    print(f"\nRecommending movies similar to '{movie_title}'...")
    recommendations = recommend_movies(movie_title, filtered_df, similarity_matrix, top_n=10)
    
    # Print recommendations
    if recommendations:
        print("\nTop 10 Recommendations:")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie['title']} (Genres: {movie['genres']}, Similarity Score: {movie['similarity_score']:.2f})")
        
        # Evaluate recommendations
        ground_truth = ["Inception", "The Matrix", "Interstellar", "The Prestige", "Memento", "Shutter Island", "The Dark Knight", "Dunkirk", "Tenet", "Source Code", "Blade Runner 2049", "Arrival", "Edge of Tomorrow", "Looper", "Predestination"]  # Example ground truth
        precision, recall = evaluate_recommendations(recommendations, ground_truth)
        diversity_score = calculate_diversity_score(recommendations)
        
        print(f"\nPrecision@10: {precision:.2f}")
        print(f"Recall@10: {recall:.2f}")
        print(f"Diversity Score: {diversity_score:.2f}")
    else:
        print("No recommendations found.")