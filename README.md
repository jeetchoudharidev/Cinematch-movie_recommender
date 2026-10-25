# CineMatch — Movie Recommendation System 🎬

A content-based movie recommender built from the **TMDB 5000 Movie Dataset** using Python and scikit-learn. Pick a movie and get five similar films based on story text, genres, keywords, cast, and director.

## What’s included

- **Streamlit frontend** with searchable movie selection, a dark cinema-inspired interface, posters, summaries, and similarity scores.
- **Content-based model** following the original notebook's approach: Porter stemming → `CountVectorizer` (up to 5,000 features) → cosine similarity.
- **Compact sparse vectors** instead of saving a full dense similarity matrix, so recommendations use less storage.
- **Automatic dataset loading** through KaggleHub for the public TMDB dataset. If automatic download is unavailable, the app can use the two CSV files placed locally.
- `RECCOMMLSYS.ipynb`: your original experimentation notebook.
- `build_model.py`: reusable preprocessing and optional artifact builder.
- `app.py`: the web application.

## Run locally

Use Python 3.11 or 3.12 for a straightforward setup.

```bash
git clone https://github.com/jeetchoudharidev/RECOMMENDATION_SYS-ML.git
cd RECOMMENDATION_SYS-ML

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first startup may take longer because the public dataset is downloaded and the sparse feature matrix is built. The app caches the model while the Streamlit process is running.

### If dataset download fails

Download the TMDB 5000 Movie Dataset and place these files in the project root or in a `data/` folder:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Then run `streamlit run app.py` again. You can also prebuild a compressed model artifact:

```bash
python build_model.py --output artifacts/movie_recommender.joblib
```

The artifact is ignored by Git by default; it is optional and not required for the app to build the model automatically.

## Deploy on Streamlit Community Cloud

1. Push the project to your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select this repository and the `main` branch.
4. Set **Main file path** to `app.py`, then deploy.
5. On the first launch, allow time for dataset download and model preparation. If the app cannot access the public dataset, add the two CSV files described above to the repository (or prebuild and include a model artifact).

Streamlit uses `requirements.txt` to install the app dependencies.

## Security note

Never put API keys or tokens in notebooks, source files, or frontend code. If a credential has ever been committed, revoke or rotate it; removing it from the latest file does not remove it from Git history.

## How recommendations work

1. Combine each movie's overview, genres, keywords, first three cast members, and director into a text profile.
2. Stem the words and convert profiles into vectors with `CountVectorizer`.
3. Compute cosine similarity between the selected movie's vector and the catalog.
4. Return the top five other titles with the highest similarity.

The similarity score is a ranking signal based on metadata, not a prediction of a user's rating or a guarantee they will enjoy the film.
