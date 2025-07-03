# Modern Streamlit Book Recommender with Dark/Light Mode Toggle and Enhanced UI
import pickle
import streamlit as st
import numpy as np

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="📚 Book Recommender", layout="wide")

# ---------------------- Theme Toggle ----------------------
toggle = st.toggle("🌗 Toggle Dark Mode")

dark_theme = {
    "bg": "#1e1e1e",
    "primary": "#00adb5",
    "text": "#eeeeee",
    "card": "#2e2e2e",
    "bg_image": "url('https://www.transparenttextures.com/patterns/black-linen.png')"
}
light_theme = {
    "bg": "#f9f9f9",
    "primary": "#4B8BBE",
    "text": "#333333",
    "card": "#ffffff",
    "bg_image": "url('https://www.transparenttextures.com/patterns/pw-maze-white.png')"
}

theme = dark_theme if toggle else light_theme

# ---------------------- Custom CSS ----------------------
st.markdown(f"""
    <style>
        html, body, [class*="css"]  {{
            background-color: {theme['bg']} !important;
            background-image: {theme['bg_image']};
            color: {theme['text']} !important;
        }}
        .title {{
            font-size:40px !important;
            font-weight:700;
            color:{theme['primary']};
            text-align:center;
            margin-bottom: 30px;
        }}
        .recommendation-box {{
            padding: 15px;
            border-radius: 10px;
            background-color: {theme['card']};
            margin-bottom: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }}
        .recommendation-box:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        }}
        .stButton>button {{
            background-color: {theme['primary']};
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 3em;
        }}
        .footer-container {{
            text-align: center;
            color: {theme['text']};
            font-size: 14px;
            padding: 20px 0;
            margin-top: 40px;
        }}
        a {{
            text-decoration: none;
            font-weight: bold;
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------- Title ----------------------
st.markdown(f'<div class="title">✨ AI Powered Book Recommender</div>', unsafe_allow_html=True)

# ---------------------- Load Artifacts ----------------------
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))  # should be a list of all titles
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

# ---------------------- Helper Functions ----------------------
def fetch_posters(suggestions):
    posters = []
    for idx in suggestions[0]:
        title = book_pivot.index[idx]
        row_idx = np.where(final_rating['title'] == title)[0][0]
        posters.append(final_rating.iloc[row_idx]['img_url'])
    return posters

def recommend_books(book_title):
    book_idx = np.where(book_pivot.index == book_title)[0][0]
    _, suggestions = model.kneighbors(book_pivot.iloc[book_idx].values.reshape(1, -1), n_neighbors=6)
    titles = [book_pivot.index[i] for i in suggestions[0]]
    posters = fetch_posters(suggestions)
    return titles, posters

# ---------------------- Input ----------------------
selected_book = st.selectbox("🔍 Search or select a book", book_names)

# ---------------------- Output ----------------------
if st.button("🎯 Show Recommendations"):
    rec_titles, rec_posters = recommend_books(selected_book)
    cols = st.columns(5)
    for i in range(1, 6):
        with cols[i - 1]:
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.image(rec_posters[i], use_container_width=True)
            st.markdown(f"<p style='text-align:center;font-weight:bold'>{rec_titles[i]}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.markdown(f"""
    <div class="footer-container">
        © 2025 Book Recommender | Crafted with ❤️ by Shivansh Tyagi<br>
        <a style='color:{theme['primary']}' href='https://github.com/Shivaanshtyagii' target='_blank'>GitHub</a> |
        <a style='color:{theme['primary']}' href='mailto:shivanshtyagi.cse@gmail.com'>Email</a>
    </div>
""", unsafe_allow_html=True)
