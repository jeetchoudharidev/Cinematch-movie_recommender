from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from build_model import load_or_build_model

st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
      :root { color-scheme: dark; }
      .stApp {
        background:
          radial-gradient(ellipse at 7% 0%, rgba(100, 75, 220, .22), transparent 36%),
          radial-gradient(ellipse at 95% 16%, rgba(36, 112, 202, .15), transparent 32%),
          #080b13;
        color: #f4f5fb;
        font-family: 'DM Sans', sans-serif;
      }
      h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
      }
      .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 3.5rem; }
      .brand-row { display:flex; align-items:center; gap:.6rem; color:#b9a9ff; font-weight:700; letter-spacing:.16em; font-size:.78rem; text-transform:uppercase; }
      .hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.2rem,5vw,4.3rem); line-height:1.03; font-weight:700; letter-spacing:-.055em; margin:.65rem 0 .7rem; }
      .hero-title span { background:linear-gradient(95deg,#c7b8ff 0%,#8ea9ff 55%,#75e1df 100%); color:transparent; background-clip:text; -webkit-background-clip:text; }
      .hero-subtitle { color:#aab2c5; font-size:1.06rem; max-width:660px; line-height:1.75; }
      .section-kicker { color:#9b8cff; text-transform:uppercase; letter-spacing:.15em; font-size:.75rem; font-weight:700; margin-bottom:.3rem; }
      .soft-divider { border:0; border-top:1px solid rgba(161,173,207,.15); margin:1.6rem 0; }
      div[data-testid="stMetric"] { background:rgba(255,255,255,.035); border:1px solid rgba(180,190,220,.12); padding:1rem 1.1rem; border-radius:16px; }
      div[data-testid="stMetricLabel"] { color:#a7afc4; }
      .movie-title { font-family:'Space Grotesk',sans-serif; font-size:1.04rem; font-weight:700; color:#f6f5ff; margin:.4rem 0 .2rem; }
      .movie-meta { color:#a6aec2; font-size:.82rem; margin-bottom:.4rem; }
      .match-pill { display:inline-block; border:1px solid rgba(157,135,255,.34); background:rgba(125,93,255,.12); color:#c9bdff; padding:.23rem .55rem; border-radius:99px; font-size:.73rem; font-weight:700; }
      .poster-placeholder { display:flex; align-items:center; justify-content:center; width:100%; aspect-ratio:2/3; border-radius:12px; background:linear-gradient(145deg,#1b2140,#211a38 55%,#14333b); font-size:2.5rem; }
      .footer { color:#707a92; font-size:.8rem; text-align:center; padding:1.8rem 0 .2rem; }
      .stButton > button[kind="primary"] { border:0; background:linear-gradient(100deg,#7861ef,#4b8fd8); color:white; font-weight:700; border-radius:12px; min-height:3rem; }
      .stButton > button[kind="primary"]:hover { border:0; filter:brightness(1.12); }
      div[data-baseweb="select"] > div { background:#101524; border-color:#2a324a; border-radius:12px; }
      .small-note { font-size:.85rem; color:#969fb5; line-height:1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the movie catalog and recommendation model...")
def get_model() -> dict[str, Any]:
    return load_or_build_model(Path("artifacts/movie_recommender.joblib"))


def poster_url(poster_path: Any) -> str | None:
    if poster_path is None:
        return None
    path = str(poster_path).strip()
    if not path or path.lower() in {"nan", "none", "null"} or not path.startswith("/"):
        return None
    return f"https://image.tmdb.org/t/p/w500{path}"


def movie_year(release_date: Any) -> str:
    value = str(release_date or "").strip()
    return value[:4] if len(value) >= 4 and value[:4].isdigit() else "Year unavailable"


def find_recommendations(model: dict[str, Any], selected_index: int, count: int = 5) -> list[tuple[int, float]]:
    movies = model["movies"]
    vectors = model["vectors"]
    similarities = cosine_similarity(vectors[selected_index], vectors).ravel()
    ranked_indices = np.argsort(similarities)[::-1]

    chosen_title = str(movies.iloc[selected_index]["title"])
    results: list[tuple[int, float]] = []
    seen_titles = {chosen_title.casefold()}
    for index in ranked_indices:
        title = str(movies.iloc[int(index)]["title"])
        if title.casefold() in seen_titles:
            continue
        seen_titles.add(title.casefold())
        results.append((int(index), float(similarities[index])))
        if len(results) >= count:
            break
    return results


def render_movie_card(movie: Any, score: float, number: int) -> None:
    with st.container(border=True):
        image = poster_url(movie.get("poster_path", ""))
        if image:
            st.image(image, use_container_width=True)
        else:
            st.markdown('<div class="poster-placeholder">🎬</div>', unsafe_allow_html=True)

        title = escape(str(movie.get("title", "Untitled")))
        st.markdown(f'<div class="movie-title">{number:02d} · {title}</div>', unsafe_allow_html=True)
        rating = movie.get("vote_average", 0)
        try:
            rating_text = f"★ {float(rating):.1f}/10" if float(rating) > 0 else "Rating unavailable"
        except (TypeError, ValueError):
            rating_text = "Rating unavailable"
        year = movie_year(movie.get("release_date", ""))
        st.markdown(f'<div class="movie-meta">{year} &nbsp;·&nbsp; {rating_text}</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="match-pill">{score:.0%} content similarity</span>', unsafe_allow_html=True)

        genres = str(movie.get("genres_display", "")).strip()
        if genres:
            st.caption(genres)
        overview = str(movie.get("overview", "") or "").strip()
        if overview and overview.lower() != "nan":
            st.write(overview[:210] + ("…" if len(overview) > 210 else ""))
        else:
            st.write("No synopsis is available for this title.")


st.markdown('<div class="brand-row">✦ CINEMATCH <span style="color:#68728b">/</span> SMART MOVIE DISCOVERY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Your next favorite film<br/>is <span>one recommendation away.</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Pick a movie you love. CineMatch looks at story, genres, keywords, cast and director to uncover films with a similar feel.</div>',
    unsafe_allow_html=True,
)

try:
    model = get_model()
except Exception as exc:
    st.error("The movie catalog could not be loaded.")
    st.markdown(
        "CineMatch can download the public TMDB dataset automatically. If that download is unavailable, "
        "place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project root or in a `data/` folder."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

movies = model["movies"]
vectors = model["vectors"]
if len(movies) == 0 or vectors.shape[0] == 0:
    st.error("The model loaded, but it contains no movies.")
    st.stop()

metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.metric("Movies in catalog", f"{len(movies):,}")
with metric2:
    st.metric("Recommendation style", "Content-based")
with metric3:
    st.metric("Signals used", "Story + credits")

st.markdown('<hr class="soft-divider"/>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">01 / Choose your starting point</div>', unsafe_allow_html=True)
st.subheader("What have you been watching lately?")

# Streamlit's selectbox is searchable. Using row indices also handles duplicate movie titles safely.
movie_indices = list(range(len(movies)))
def option_label(index: int) -> str:
    row = movies.iloc[index]
    return f"{row['title']} ({movie_year(row.get('release_date', ''))})"

selected_index = st.selectbox(
    "Search the movie catalog",
    options=movie_indices,
    format_func=option_label,
    index=None,
    placeholder="Start typing a movie title…",
    label_visibility="collapsed",
)

button_col, helper_col = st.columns([1, 2])
with button_col:
    recommend_clicked = st.button("✦ Find similar movies", type="primary", use_container_width=True, disabled=selected_index is None)
with helper_col:
    st.markdown('<p class="small-note">Recommendations are generated from movie content, not personal watch history. Search by title, then explore the closest matches.</p>', unsafe_allow_html=True)

if recommend_clicked and selected_index is not None:
    st.session_state["cine_selected_index"] = int(selected_index)
    st.session_state["cine_recommendations"] = find_recommendations(model, int(selected_index), count=5)

if "cine_recommendations" in st.session_state:
    active_index = st.session_state.get("cine_selected_index")
    if active_index is not None:
        selected_movie = movies.iloc[int(active_index)]
        st.markdown('<hr class="soft-divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">02 / Your movie matches</div>', unsafe_allow_html=True)
        st.subheader(f"Because you picked {selected_movie['title']}")
        source_genres = str(selected_movie.get("genres_display", "")).strip()
        if source_genres:
            st.caption(f"Starting genre profile: {source_genres}")

        recommendations = st.session_state["cine_recommendations"]
        if not recommendations:
            st.info("There were no similar movies to show for this selection.")
        else:
            columns = st.columns(len(recommendations), gap="medium")
            for position, ((movie_index, score), column) in enumerate(zip(recommendations, columns), start=1):
                with column:
                    render_movie_card(movies.iloc[movie_index], score, position)

st.markdown(
    '<div class="footer">Built with Python · scikit-learn · Streamlit &nbsp;|&nbsp; Content similarity is a ranking signal, not a rating prediction.</div>',
    unsafe_allow_html=True,
)
