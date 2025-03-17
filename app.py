import streamlit as st
import pandas as pd
from scripts.train_model import load_engineered_dataset, filter_dataset, safe_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

# Set page configuration
st.set_page_config(
    page_title="What2Watch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix-inspired design
st.markdown(
    """
    <style>
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background-color: #333333;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #E50914;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #B20710;
    }
    .stSidebar {
        background-color: #000000;
    }
    .stMarkdown h1 {
        color: #E50914;
    }
    .stMarkdown h2 {
        color: #E50914;
    }
    .stMarkdown h3 {
        color: #E50914;
    }
    .stMarkdown p {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.title("🎬 What2Watch")
st.markdown("Discover movies similar to your favorites! 🍿")

# Load and preprocess the dataset
@st.cache_data  # Cache the dataset to improve performance
def load_data(min_votes, min_year):
    """
    Load and preprocess the dataset.
    """
    engineered_filepath = 'data/engineered_movies.csv'
    engineered_df = load_engineered_dataset(engineered_filepath)
    filtered_df = filter_dataset(engineered_df, min_votes=min_votes, min_year=min_year)
    
    # Process textual data separately (overview, tagline, production_companies)
    text_features = filtered_df[['overview', 'tagline', 'production_companies']].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Vectorize textual features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_features=2000)  # Reduce features to save memory
    tfidf_matrix = vectorizer.fit_transform(text_features)

    # Convert genres and keywords into a one-hot encoded matrix
    mlb = MultiLabelBinarizer(sparse_output=True)  # Use sparse matrices

    # Convert stringified lists into real lists
    filtered_df['genres'] = filtered_df['genres'].apply(safe_eval)  # Convert genres from string to list
    filtered_df['keywords'] = filtered_df['keywords'].apply(safe_eval)  # Convert keywords from string to list

    genres_matrix = mlb.fit_transform(filtered_df['genres'])
    keywords_matrix = mlb.fit_transform(filtered_df['keywords'])

    # Combine text embeddings with categorical embeddings
    combined_features_matrix = hstack([tfidf_matrix, genres_matrix, keywords_matrix])

    return filtered_df, combined_features_matrix

# Sidebar for filters
st.sidebar.title("🎬 Filters")
min_votes = st.sidebar.slider("Minimum Votes", min_value=0, max_value=1000, value=50, help="Filter movies by minimum number of votes.")
min_year = st.sidebar.slider("Minimum Release Year", min_value=1980, max_value=2025, value=1980, help="Filter movies by release year.")

# Load data with filters
filtered_df, combined_features_matrix = load_data(min_votes, min_year)

# User input with autocomplete
movie_titles = filtered_df['title'].tolist()
movie_title = st.selectbox("Enter a movie title:", movie_titles, index=None, placeholder="e.g., Inception")

# Get recommendations
if st.button("Get Recommendations"):
    if movie_title:
        # Case-insensitive search and fuzzy matching
        matches = process.extract(movie_title, filtered_df['title'], limit=5)  # Get top 5 matches
        
        if matches:
            # Use the best match
            best_match, score, _ = matches[0]
            if score >= 80:  # Only use matches with a score of 80 or higher
                st.success(f"Did you mean '{best_match}'? Here are similar movies:")
                
                # Find the index of the selected movie
                movie_index = filtered_df[filtered_df['title'] == best_match].index[0]
                
                # Compute similarities on-the-fly
                movie_similarities = cosine_similarity(combined_features_matrix[movie_index], combined_features_matrix).flatten()
                
                # Get top 10 similar movies (excluding the movie itself)
                similar_indices = movie_similarities.argsort()[::-1][1:11]
                recommendations = filtered_df.iloc[similar_indices][['title', 'genres', 'poster_path']]
                
                # Display recommendations
                for i, movie in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"### {i}. **{movie['title']}**")
                        st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
                        st.markdown(f"**Similarity Score:** {movie_similarities[i]:.2f}")
                        
                        # Display movie poster if available
                        if movie['poster_path']:
                            st.image(f"https://image.tmdb.org/t/p/w500{movie['poster_path']}", width=200)
                        
                        st.markdown("---")  # Add a separator between movies
            else:
                st.error(f"Sorry, we couldn't find a close match for '{movie_title}'. Please try again.")
        else:
            st.error(f"Sorry, '{movie_title}' is not in our database. Please try another movie.")
    else:
        st.warning("Please enter a movie title.")

# Footer
st.markdown("---")
st.markdown("### About What2Watch")
st.markdown("What2Watch is a movie recommendation system that helps you discover movies similar to your favorites. Built with ❤️ using Python and Streamlit.")
st.markdown("**GitHub Repository:** [What2Watch](https://github.com/dafoakwah/what2watch)")