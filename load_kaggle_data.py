# scripts/load_kaggle_data.py
import pandas as pd

def load_kaggle_dataset(filepath):
    """
    Load the Kaggle dataset and clean it.
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Inspect the 'genres' column
    print("Sample 'genres' column values:")
    print(df['genres'].head())
    
    # Clean the data
    # 1. Convert 'release_date' to datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # 2. Convert 'genres' from comma-separated string to list
    df['genres'] = df['genres'].fillna('').apply(lambda x: x.split(', ') if x else [])
    
    # 3. Handle missing values (example: fill missing 'overview' with empty string)
    df['overview'] = df['overview'].fillna('')
    
    # 4. Drop rows with missing critical data (e.g., 'title' or 'id')
    df = df.dropna(subset=['title', 'id'])
    
    return df

if __name__ == '__main__':
    # Path to the Kaggle dataset
    kaggle_filepath = 'data/kaggle_movies.csv'  # Updated file path
    
    # Load the dataset
    print("Loading Kaggle dataset...")
    kaggle_df = load_kaggle_dataset(kaggle_filepath)
    
    # Print summary
    print(f"Loaded {len(kaggle_df)} movies from Kaggle dataset.")
    print("\nFirst 5 rows of the dataset:")
    print(kaggle_df.head())
    
    # Save the cleaned dataset to a new CSV (optional)
    output_filepath = 'data/cleaned_kaggle_movies.csv'
    kaggle_df.to_csv(output_filepath, index=False)
    print(f"\nCleaned dataset saved to {output_filepath}")