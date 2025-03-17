# scripts/feature_engineering.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_cleaned_dataset(filepath):
    """
    Load the cleaned dataset with optimized data types.
    """
    # Specify data types for problematic columns
    dtype = {
        'release_date': 'str'  # Treat as string to avoid mixed types
    }
    
    # Load the dataset
    df = pd.read_csv(filepath, dtype=dtype, low_memory=False)
    return df

def calculate_weighted_rating(df):
    """
    Calculate weighted ratings using IMDB's formula.
    """
    # IMDB's formula: (v / (v + m)) * R + (m / (v + m)) * C
    # Where:
    # - R = vote_average
    # - v = vote_count
    # - m = minimum votes required to be listed (e.g., 90th percentile)
    # - C = mean vote_average across the dataset
    
    # Calculate mean vote_average (C)
    C = df['vote_average'].mean()
    
    # Calculate minimum votes required (m)
    m = df['vote_count'].quantile(0.9)
    
    # Filter movies with at least m votes
    qualified_movies = df[df['vote_count'] >= m].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Calculate weighted ratings
    qualified_movies.loc[:, 'weighted_rating'] = (
        (qualified_movies['vote_count'] / (qualified_movies['vote_count'] + m)) * qualified_movies['vote_average'] +
        (m / (qualified_movies['vote_count'] + m)) * C
    )
    
    return qualified_movies

def extract_release_year(df):
    """
    Extract the release year from the release_date column.
    """
    df.loc[:, 'release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    return df

def create_genre_embeddings(df):
    """
    Convert genres into numerical embeddings.
    """
    # Create a binary matrix for genres
    genres = df['genres'].apply(lambda x: '|'.join(x))  # Convert list of genres to string
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
    genre_embeddings = vectorizer.fit_transform(genres)
    
    # Convert to DataFrame
    genre_embeddings_df = pd.DataFrame(genre_embeddings.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Add genre embeddings to the original dataset
    df = pd.concat([df, genre_embeddings_df], axis=1)
    
    return df

def save_engineered_dataset(df, output_filepath):
    """
    Save the engineered dataset to a CSV file.
    """
    df.to_csv(output_filepath, index=False)
    print(f"Engineered dataset saved to {output_filepath}")

if __name__ == '__main__':
    # Load the cleaned dataset
    cleaned_filepath = 'data/cleaned_movies.csv'
    print("Loading cleaned dataset...")
    cleaned_df = load_cleaned_dataset(cleaned_filepath)
    
    # Feature engineering
    print("Performing feature engineering...")
    
    # Calculate weighted ratings
    print("Calculating weighted ratings...")
    weighted_df = calculate_weighted_rating(cleaned_df)
    
    # Extract release year
    print("Extracting release year...")
    weighted_df = extract_release_year(weighted_df)
    
    # Create genre embeddings
    print("Creating genre embeddings...")
    engineered_df = create_genre_embeddings(weighted_df)
    
    # Save the engineered dataset
    output_filepath = 'data/engineered_movies.csv'
    save_engineered_dataset(engineered_df, output_filepath)