import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TMDB API configuration
API_KEY = os.getenv('TMDB_API_KEY')  # Read API key from .env file
BASE_URL = 'https://api.themoviedb.org/3'

def fetch_all_movies(api_key, max_pages=500):  # Limit to 500 pages (10,000 movies)
    all_movies = []
    url = f'{BASE_URL}/movie/popular'
    
    for page in range(1, max_pages + 1):
        params = {'api_key': api_key, 'language': 'en-US', 'page': page}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            all_movies.extend(data['results'])
            print(f"Fetched page {page} of {max_pages}")
        else:
            print(f"Error fetching page {page}: {response.status_code}")
            break
        
        time.sleep(0.25)  # Add a delay to avoid rate limits
    
    return all_movies

def standardize_tmdb_data(tmdb_movies):
    """
    Standardize TMDB API data to match Kaggle dataset format.
    """
    tmdb_df = pd.DataFrame(tmdb_movies)
    
    # Map TMDB columns to Kaggle columns
    tmdb_df = tmdb_df.rename(columns={
        'id': 'id',
        'title': 'title',
        'vote_average': 'vote_average',
        'vote_count': 'vote_count',
        'release_date': 'release_date',
        'overview': 'overview',
        'popularity': 'popularity',
        'poster_path': 'poster_path',
        'backdrop_path': 'backdrop_path',
        'original_language': 'original_language',
        'original_title': 'original_title',
        'adult': 'adult',
        'genre_ids': 'genres'
    })
    
    # Convert release_date to datetime
    tmdb_df['release_date'] = pd.to_datetime(tmdb_df['release_date'], errors='coerce')
    
    # Map genre IDs to genre names
    genre_mapping = fetch_genre_mapping(API_KEY)
    tmdb_df['genres'] = tmdb_df['genres'].apply(
        lambda genre_ids: [genre_mapping.get(genre_id, 'Unknown') for genre_id in genre_ids]
    )
    
    return tmdb_df

def fetch_genre_mapping(api_key):
    """
    Fetch genre mapping from TMDB API.
    """
    url = f'{BASE_URL}/genre/movie/list'
    params = {'api_key': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        genres = response.json()['genres']
        return {genre['id']: genre['name'] for genre in genres}
    return {}

if __name__ == '__main__':
    # Fetch data from TMDB API
    print("Fetching data from TMDB API...")
    tmdb_movies = fetch_all_movies(API_KEY)
    
    # Standardize TMDB data
    print("Standardizing TMDB data...")
    tmdb_df = standardize_tmdb_data(tmdb_movies)
    
    # Print summary
    print(f"Fetched {len(tmdb_df)} movies from TMDB API.")
    print("\nFirst 5 rows of TMDB data:")
    print(tmdb_df.head())
    
    # Save TMDB data to a CSV file
    output_filepath = 'data/tmdb_movies.csv'
    tmdb_df.to_csv(output_filepath, index=False)
    print(f"\nTMDB data saved to {output_filepath}")