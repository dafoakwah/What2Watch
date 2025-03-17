# scripts/combine_datasets.py
import pandas as pd

def load_datasets(kaggle_filepath, tmdb_filepath):
    """
    Load the Kaggle and TMDB datasets.
    """
    print("Loading Kaggle dataset...")
    kaggle_df = pd.read_csv(kaggle_filepath)
    
    print("Loading TMDB dataset...")
    tmdb_df = pd.read_csv(tmdb_filepath)
    
    return kaggle_df, tmdb_df

def combine_datasets(kaggle_df, tmdb_df):
    """
    Combine Kaggle and TMDB datasets, ensuring no duplicates.
    """
    # Add a source column to track where each movie came from
    kaggle_df['source'] = 'kaggle'
    tmdb_df['source'] = 'tmdb'
    
    # Combine datasets
    combined_df = pd.concat([kaggle_df, tmdb_df], ignore_index=True)
    
    # Drop duplicates based on 'id' (prefer TMDB data for newer movies)
    combined_df = combined_df.sort_values(by='release_date', ascending=False)
    combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')
    
    return combined_df

def save_combined_dataset(combined_df, output_filepath):
    """
    Save the combined dataset to a CSV file.
    """
    combined_df.to_csv(output_filepath, index=False)
    print(f"Combined dataset saved to {output_filepath}")

if __name__ == '__main__':
    # File paths
    kaggle_filepath = 'data/cleaned_kaggle_movies.csv'
    tmdb_filepath = 'data/tmdb_movies.csv'
    output_filepath = 'data/combined_movies.csv'
    
    # Load datasets
    kaggle_df, tmdb_df = load_datasets(kaggle_filepath, tmdb_filepath)
    
    # Combine datasets
    print("Combining datasets...")
    combined_df = combine_datasets(kaggle_df, tmdb_df)
    
    # Print summary
    print(f"Combined dataset has {len(combined_df)} movies.")
    print("\nFirst 5 rows of the combined dataset:")
    print(combined_df.head())
    
    # Save the combined dataset
    save_combined_dataset(combined_df, output_filepath)