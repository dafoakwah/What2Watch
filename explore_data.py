# scripts/explore_data.py
import pandas as pd
import matplotlib.pyplot as plt

def load_combined_dataset(filepath):
    """
    Load the combined dataset.
    """
    df = pd.read_csv(filepath)
    return df

def explore_data(df):
    """
    Perform exploratory data analysis (EDA) on the combined dataset.
    """
    # Basic info
    print("Dataset Info:")
    print(df.info())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Count of movies by source
    print("\nCount of Movies by Source:")
    print(df['source'].value_counts())
    
    # Distribution of vote_average
    plt.figure(figsize=(10, 6))
    df['vote_average'].hist(bins=30, edgecolor='black')
    plt.title('Distribution of Vote Average')
    plt.xlabel('Vote Average')
    plt.ylabel('Number of Movies')
    plt.show()
    
    # Top 10 genres
    genres = df['genres'].explode()
    print("\nTop 10 Genres:")
    print(genres.value_counts().head(10))

if __name__ == '__main__':
    # Load the combined dataset
    combined_filepath = 'data/combined_movies.csv'
    print("Loading combined dataset...")
    combined_df = load_combined_dataset(combined_filepath)
    
    # Perform EDA
    print("Performing exploratory data analysis...")
    explore_data(combined_df)