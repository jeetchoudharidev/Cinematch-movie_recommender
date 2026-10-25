# 🎬 CineMatch — Movie Recommendation System

A content-based movie recommendation system built with Python and scikit-learn using the TMDB 5000 movie dataset.

## Features
- Search/select a movie and discover similar titles
- Content-based recommendations from overview, genres, keywords, cast, and director
- Cosine similarity over CountVectorizer features
- Streamlit UI with poster art from TMDB image URLs when configured
- Graceful poster fallback when no image is available

## Run locally
1. Clone the repo.
2. Ensure the generated model files `movies.pkl` and `similarity.pkl` are present in the project root.
3. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model artifacts
The current notebook saves `movies.pkl`, `movie_dict.pkl`, and `similarity.pkl`. These generated files are not currently committed to the repository. Run the notebook through the preprocessing/vectorization/recommendation sections and save the artifacts in the repository root, or update `app.py` to match your artifact locations.

**Important:** never commit API credentials. If a credential was ever committed to a notebook or repository, revoke it and rotate it.
