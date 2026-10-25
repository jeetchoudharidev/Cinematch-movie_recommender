"""Build a compact, deployable movie-recommendation model.

The model follows the original notebook's content-based approach:
overview + genres + keywords + top cast + director -> stemming ->
CountVectorizer -> cosine similarity at request time.

Dataset CSVs are read locally when present. Otherwise, the public TMDB 5000
metadata dataset is fetched through KaggleHub.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

DATASET_ID = "tmdb/tmdb-movie-metadata"
MOVIES_FILENAME = "tmdb_5000_movies.csv"
CREDITS_FILENAME = "tmdb_5000_credits.csv"
DEFAULT_ARTIFACT = Path("artifacts/movie_recommender.joblib")
_STEMMER = PorterStemmer()


def _find_csvs() -> tuple[Path, Path]:
    """Find the TMDB CSV files locally, or download the public dataset."""
    search_roots = [Path.cwd(), Path.cwd() / "data"]
    for root in search_roots:
        movies_path = root / MOVIES_FILENAME
        credits_path = root / CREDITS_FILENAME
        if movies_path.is_file() and credits_path.is_file():
            return movies_path, credits_path

    try:
        import kagglehub

        dataset_dir = Path(kagglehub.dataset_download(DATASET_ID))
        movies_path = dataset_dir / MOVIES_FILENAME
        credits_path = dataset_dir / CREDITS_FILENAME
        if movies_path.is_file() and credits_path.is_file():
            return movies_path, credits_path
    except Exception as exc:
        raise RuntimeError(
            "Could not download the public TMDB dataset. Add "
            f"{MOVIES_FILENAME} and {CREDITS_FILENAME} to the project root "
            "or the ./data folder, then restart the app. Original error: "
            f"{exc}"
        ) from exc

    raise FileNotFoundError(
        f"Dataset files were not found. Expected {MOVIES_FILENAME} and "
        f"{CREDITS_FILENAME} in the project root or ./data folder."
    )


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    """Safely parse the dataset's JSON-like list columns."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def _names(value: Any, limit: int | None = None) -> list[str]:
    items = _parse_json_list(value)
    names = [str(item.get("name", "")).strip() for item in items
             if isinstance(item, dict) and item.get("name")]
    return names if limit is None else names[:limit]


def _director(value: Any) -> list[str]:
    for item in _parse_json_list(value):
        if isinstance(item, dict) and item.get("job") == "Director" and item.get("name"):
            return [str(item["name"]).strip()]
    return []


def _stem_words(value: str) -> str:
    return " ".join(_STEMMER.stem(word) for word in value.split())


def build_movie_model() -> dict[str, Any]:
    """Load data and create a sparse feature matrix similar to the notebook."""
    movies_csv, credits_csv = _find_csvs()
    raw_movies = pd.read_csv(movies_csv)
    raw_credits = pd.read_csv(credits_csv)

    needed = ["id", "title", "overview", "genres", "keywords"]
    for col in needed:
        if col not in raw_movies.columns:
            raise ValueError(f"Dataset movies file is missing the '{col}' column.")

    optional_columns = [
        col for col in ["poster_path", "release_date", "vote_average"]
        if col in raw_movies.columns
    ]
    movie_cols = needed + optional_columns
    movies = raw_movies[movie_cols].copy()

    credit_cols = ["movie_id", "cast", "crew"]
    missing_credit_cols = [col for col in credit_cols if col not in raw_credits.columns]
    if missing_credit_cols:
        raise ValueError(f"Credits file is missing columns: {missing_credit_cols}")

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    raw_credits["movie_id"] = pd.to_numeric(raw_credits["movie_id"], errors="coerce")
    merged = movies.merge(
        raw_credits[credit_cols],
        left_on="id",
        right_on="movie_id",
        how="inner",
    )
    merged = merged.dropna(subset=["id", "title"]).copy()
    merged["title"] = merged["title"].astype(str).str.strip()
    merged = merged[merged["title"] != ""].reset_index(drop=True)

    overviews: list[str] = []
    genres_display: list[str] = []
    tags: list[str] = []
    for _, row in merged.iterrows():
        overview = str(row.get("overview", "") or "")
        genre_names = _names(row.get("genres"))
        keyword_names = _names(row.get("keywords"))
        cast_names = _names(row.get("cast"), limit=3)
        director_names = _director(row.get("crew"))

        # Keep feature-name spaces removed, matching the original notebook.
        compact_features = [
            item.replace(" ", "")
            for item in genre_names + keyword_names + cast_names + director_names
        ]
        combined = " ".join([overview, *compact_features]).strip().lower()
        overviews.append(overview if overview.lower() != "nan" else "")
        genres_display.append(", ".join(genre_names[:3]))
        tags.append(_stem_words(combined))

    merged["overview_display"] = overviews
    merged["genres_display"] = genres_display
    merged["tags"] = tags
    merged["release_date"] = merged.get("release_date", pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["poster_path"] = merged.get("poster_path", pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["vote_average"] = pd.to_numeric(
        merged.get("vote_average", pd.Series(0.0, index=merged.index)),
        errors="coerce",
    ).fillna(0.0)

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(merged["tags"].fillna(""))

    metadata = merged[
        ["id", "title", "overview_display", "genres_display", "poster_path",
         "release_date", "vote_average", "tags"]
    ].rename(columns={"overview_display": "overview"}).reset_index(drop=True)

    return {
        "model_version": 1,
        "movies": metadata,
        "vectors": vectors,
    }


def load_or_build_model(artifact_path: str | Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    """Load a prebuilt artifact if available; otherwise build from the dataset."""
    path = Path(artifact_path)
    if path.is_file():
        model = joblib.load(path)
        if isinstance(model, dict) and "movies" in model and "vectors" in model:
            return model
        raise ValueError(f"The model artifact at {path} has an unexpected format.")
    return build_movie_model()


def save_model(model: dict[str, Any], output_path: str | Path = DEFAULT_ARTIFACT) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a compact movie recommender artifact.")
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT), help="Output joblib file path.")
    args = parser.parse_args()
    print("Preparing movie data and content vectors...")
    model_data = build_movie_model()
    output = save_model(model_data, args.output)
    print(f"Saved {len(model_data['movies']):,} movies to {output}")
