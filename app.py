import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, SVD

# Load movie data
df = pd.read_csv('movies.csv')

ratings_df =pd.read_csv('ratings.csv')


# Content-Based Filtering
df['genres'] = df['genres'].str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])
content_model = NearestNeighbors(n_neighbors=3, metric='cosine')
content_model.fit(tfidf_matrix)

# Collaborative Filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
collab_model = SVD()
collab_model.fit(trainset)

# Enhanced UI
st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        text-align: center;
        color: #ff4b4b;
    }
    .header {
        font-size: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown('<div class="title">🎥 Hybrid Movie Recommendation System 🎬</div>', unsafe_allow_html=True)
st.write("This app provides movie recommendations using a hybrid approach (content-based + collaborative filtering).")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Content-Based Recommendations", "Collaborative Filtering", "Hybrid Recommendations", "Top Rated Movies"])

# Content-Based Recommendations
if page == "Content-Based Recommendations":
    st.markdown('<div class="header">Content-Based Recommendations</div>', unsafe_allow_html=True)
    movie_title = st.selectbox("Select a Movie", df['title'])
    if st.button("Recommend Similar Movies"):
        idx = df.index[df['title'] == movie_title].tolist()[0]
        distances, indices = content_model.kneighbors(tfidf_matrix[idx], n_neighbors=3)
        st.write(f"Movies similar to: **{movie_title}**")
        st.table(df['title'].iloc[indices.flatten()])

# Collaborative Filtering
elif page == "Collaborative Filtering":
    st.markdown('<div class="header">Collaborative Filtering</div>', unsafe_allow_html=True)
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    movie_id = st.number_input("Enter Movie ID to Predict Rating:", min_value=1, step=1)
    if st.button("Predict Rating"):
        prediction = collab_model.predict(user_id, movie_id)
        st.write(f"Predicted Rating for User {user_id} on Movie ID {movie_id}: **{prediction.est:.2f}**")

# Hybrid Recommendations
elif page == "Hybrid Recommendations":
    st.markdown('<div class="header">Hybrid Recommendations</div>', unsafe_allow_html=True)
    movie_title = st.selectbox("Select a Movie for Hybrid Recommendations", df['title'], key="hybrid")
    user_id = st.number_input("Enter User ID for Hybrid Recommendations:", min_value=1, step=1, key="user_hybrid")
    if st.button("Get Hybrid Recommendations"):
        idx = df.index[df['title'] == movie_title].tolist()[0]
        _, content_indices = content_model.kneighbors(tfidf_matrix[idx], n_neighbors=3)
        content_recommended_movies = df['movieId'].iloc[content_indices.flatten()].tolist()

        hybrid_recommendations = []
        for mid in content_recommended_movies:
            pred = collab_model.predict(user_id, mid)
            hybrid_recommendations.append((mid, pred.est))

        hybrid_recommendations = sorted(hybrid_recommendations, key=lambda x: x[1], reverse=True)
        st.write(f"Hybrid Recommendations for **{movie_title}**:")
        for movie_id, rating in hybrid_recommendations:
            movie_name = df[df['movieId'] == movie_id]['title'].values[0]
            st.write(f"**{movie_name}** (Predicted Rating: {rating:.2f})")

# Top Rated Movies
elif page == "Top Rated Movies":
    st.markdown('<div class="header">Top Rated Movies</div>', unsafe_allow_html=True)
    top_movies = ratings_df.groupby('movieId').mean()['rating'].sort_values(ascending=False)
    top_movies_df = top_movies.head(5).reset_index()
    top_movies_df['title'] = top_movies_df['movieId'].apply(lambda x: df[df['movieId'] == x]['title'].values[0])
    st.table(top_movies_df[['title', 'rating']])
