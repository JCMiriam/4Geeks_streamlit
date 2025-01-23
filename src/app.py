import os
import json
import pandas as pd
from joblib import load
from rapidfuzz.fuzz import ratio
from rapidfuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Lazy loading for datasets
def load_movies_data():
    return pd.read_csv('../data/raw/movies_data.csv', usecols=["id", "title", "genres", "overview", "release_date", "vote_average", "runtime", "poster"])

def load_processed_movies_data():
    return pd.read_csv("../data/processed/processed_movies.csv", usecols=["title", "tags"])

# Load models
try:
    vec_model = load("../models/vec_model.joblib")
    knn_model = load("../models/knn_model.joblib")
    vectors = load("../models/vectors.joblib")
except FileNotFoundError:
    # If models are not found, create and save them
    processed_movies_df = load_processed_movies_data()
    vec_model = TfidfVectorizer(token_pattern=r'\b\w+\b', lowercase=True)
    vectors = vec_model.fit_transform(processed_movies_df["tags"])
    knn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
    knn_model.fit(vectors)

# Recommend functions
def get_movie_info_by_title(title):
    movies_df = load_movies_data()
    best_match = process.extractOne(
        title,
        movies_df["title"].values,
        scorer=ratio,
        score_cutoff=70
    )

    if not best_match:
        return None

    best_title = best_match[0]
    movie_row = movies_df[movies_df["title"] == best_title]
    movie_data = movie_row.iloc[0]
    genres = json.loads(movie_data["genres"])
    genres_list = [genre["name"] for genre in genres]

    return {
        "title": movie_data["title"],
        "genres": ", ".join(genres_list),
        "overview": movie_data["overview"],
        "release_date": movie_data.get("release_date", "Unknown release date"),
        "vote_average": movie_data["vote_average"],
        "runtime": movie_data.get("runtime", "Not available"),
        "poster_url": movie_data["poster"],
    }

def recommend(movie_title, n_recommendations=5):
    processed_movies_df = load_processed_movies_data()
    best_match = process.extractOne(
        movie_title,
        processed_movies_df["title"].values,
        scorer=ratio,
        score_cutoff=70
    )

    if not best_match:
        return []

    best_title = best_match[0]
    movie_index = processed_movies_df[processed_movies_df["title"] == best_title].index[0]
    movie_tags = processed_movies_df.loc[movie_index, "tags"]
    input_vector = vec_model.transform([movie_tags])

    distances, indices = knn_model.kneighbors(input_vector)

    recommendations = []
    for idx in indices[0]:
        recommended_title = processed_movies_df.iloc[idx]["title"]
        if recommended_title == best_title:
            continue

        movie_info = get_movie_info_by_title(recommended_title)
        if movie_info:
            recommendations.append(movie_info)

        if len(recommendations) >= n_recommendations:
            break

    return recommendations

# Streamlit interface
st.markdown("""<h3 style="padding: 0">Grab your popcorn</h3>""", unsafe_allow_html=True,)
st.markdown("""<h1 style="padding: 0; margin-bottom: 24px;">What would you watch tonight?</h1>""", unsafe_allow_html=True,)

movie_title = st.text_input("Tell us a movie you like:")

if movie_title:
    movie_info = get_movie_info_by_title(movie_title)

    if movie_info:
        poster_url = movie_info['poster_url']
        title = movie_info['title']
        genres = movie_info['genres']
        overview = movie_info['overview']
        release_date = movie_info['release_date']
        vote_average = movie_info['vote_average']

        st.subheader("If you liked")
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 40px;">
                <div>
                    <img src="{poster_url}" alt="Poster" style="width: 200px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
                </div>
                <div style="flex: 1;">
                    <h4 style="margin: 0;">{title} <span style="font-size: 16px; color: gray;">({release_date})</span></h4>
                    <p style="margin: 5px 0; font-size: 14px; color: #888;">{genres}</p>
                    <p style="display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis; font-size: 14px; color: #555; margin: 10px 0;">{overview}</p>
                    <span style="font-size: 14px; color: #333; font-weight: bold;">Rating: {vote_average}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,  # Permite interpretar HTML
        )


        st.subheader("We recommend you")
        recommendations = recommend(movie_title)

        if recommendations:
            for rec in recommendations:
                poster_url = rec['poster_url']
                title = rec['title']
                genres = rec['genres']
                overview = rec['overview']
                release_date = rec['release_date']
                vote_average = rec['vote_average']

                st.markdown(
                    f"""
                    <div style="display: flex; gap: 20px; align-items: flex-start; margin-bottom: 40px;">
                        <div style="width: 200px; height: fit-content; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); overflow: hidden;">
                            <img src="{poster_url}" alt="Poster" style="min-width: 100%; ">
                        </div>
                        <div style="flex: 1;">
                            <h4 style="margin: 0;">{title} <span style="font-size: 16px; color: gray;">({release_date})</span></h4>
                            <p style="margin: 5px 0; font-size: 14px; color: #888;">{genres}</p>
                            <p style="display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis; font-size: 14px; color: #555; margin: 10px 0;">{overview}</p>
                            <span style="font-size: 14px; color: #333; font-weight: bold;">Rating: {vote_average}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,  # Permite interpretar HTML
                )
        else:
            st.write("No recommendations found.")
