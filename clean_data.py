# scripts/clean_data.py
import pandas as pd

def load_combined_dataset(filepath):
    """
    Load the combined dataset with optimized data types.
    """
    # Specify data types for problematic columns
    dtype = {
        'release_date': 'str',  # Treat as string to avoid mixed types
        'video': 'str'          # Treat as string to avoid mixed types
    }
    
    # Load the dataset
    df = pd.read_csv(filepath, dtype=dtype, low_memory=False)
    
    return df

def clean_data(df):
    """
    Clean the combined dataset.
    """
    # Handle missing data
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')  # Convert to datetime
    df['video'] = df['video'].fillna('False')  # Fill missing 'video' values
    
    # Fill missing values with defaults
    df['backdrop_path'] = df['backdrop_path'].fillna('')
    df['homepage'] = df['homepage'].fillna('')
    df['imdb_id'] = df['imdb_id'].fillna('')
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    df['poster_path'] = df['poster_path'].fillna('')
    df['production_companies'] = df['production_companies'].fillna('[]')
    df['production_countries'] = df['production_countries'].fillna('[]')
    df['spoken_languages'] = df['spoken_languages'].fillna('[]')
    df['keywords'] = df['keywords'].fillna('[]')
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['title', 'id'])
    
    # Convert 'genres' from string to list
    df['genres'] = df['genres'].apply(lambda x: eval(x) if pd.notnull(x) else [])
    
    # Remove rows with empty genres
    df = df[df['genres'].apply(len) > 0]
    
    return df

if __name__ == '__main__':
    # Load the combined dataset
    combined_filepath = 'data/combined_movies.csv'
    print("Loading combined dataset...")
    combined_df = load_combined_dataset(combined_filepath)
    
    # Clean the dataset
    print("Cleaning dataset...")
    cleaned_df = clean_data(combined_df)
    
    # Save the cleaned dataset
    output_filepath = 'data/cleaned_movies.csv'
    cleaned_df.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset saved to {output_filepath}")