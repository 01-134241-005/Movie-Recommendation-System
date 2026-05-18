"""Machine learning model for movie recommendations."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from preprocess import MovieDataLoader, MoviePreprocessor


class RecommendationModel:
    """Base class that stores common movie data."""

    def __init__(self):
        self.movies = pd.DataFrame()
        self.is_trained = False


class KNNRecommender(RecommendationModel):
    """TF-IDF vectorizer plus a KNN recommendation engine."""

    def __init__(self, n_neighbors: int = 11):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=12000)
        self.knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
        self.tfidf_matrix = None

    def train(self) -> None:
        """Train the only recommendation model used by the project: KNN."""
        loader = MovieDataLoader()
        preprocessor = MoviePreprocessor()
        self.movies = preprocessor.preprocess(loader.load_data()).reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["clean_text"])
        self.knn_model.fit(self.tfidf_matrix)
        self._build_title_index()
        self.is_trained = True

    def _build_title_index(self) -> None:
        """Pre-build a lightweight list for instant autocomplete."""
        self._title_index = []
        for _, row in self.movies.iterrows():
            self._title_index.append(
                {
                    "title": row["title"],
                    "title_lower": str(row["title"]).lower(),
                    "poster": row["poster"],
                    "release_year": row["release_year"],
                    "genre": row["genre"],
                    "rating": row["rating"],
                }
            )

    def recommend(self, title: str, top_n: int = 10, filters: Dict[str, str] | None = None) -> List[dict]:
        """Return movies most similar to the selected title."""
        self._require_training()
        movie_index = self.find_best_match_index(title)
        filters = filters or {}
        active_filters = any(str(filters.get(key, "")).strip() for key in ("genre", "region", "language", "year", "rating"))
        pool_size = min(top_n + 1, len(self.movies))
        if active_filters:
            pool_size = min(len(self.movies), max(250, top_n * 40))
        distances, indices = self.knn_model.kneighbors(
            self.tfidf_matrix[movie_index],
            n_neighbors=pool_size,
        )
        selected = self.movies.iloc[movie_index]
        candidates = []
        for distance, index in zip(distances.flatten(), indices.flatten()):
            if index == movie_index:
                continue
            movie = self.movies.iloc[index].to_dict()
            if not self._passes_filters(movie, filters):
                continue
            similarity = round((1 - float(distance)) * 100, 2)
            boost = self._filter_alignment_boost(selected, movie, filters)
            candidates.append((similarity + boost, movie, similarity))
        candidates.sort(key=lambda item: item[0], reverse=True)
        recommendations = []
        for _score, movie, similarity in candidates[:top_n]:
            movie = dict(movie)
            movie["similarity"] = similarity
            recommendations.append(movie)
        return recommendations

    def find_best_match_index(self, query: str) -> int:
        """Find an exact title match or the nearest KNN text-vector match."""
        query = str(query).strip().lower()
        titles = self.movies["title"].str.lower()
        exact = self.movies[titles == query]
        if not exact.empty:
            return int(exact.index[0])
        query_vector = self.vectorizer.transform([query])
        _distances, indices = self.knn_model.kneighbors(query_vector, n_neighbors=1)
        return int(indices.flatten()[0])

    def search_titles(self, query: str, limit: int = 8) -> List[dict]:
        """Return live search suggestions for the autocomplete API."""
        self._require_training()
        query = str(query).strip().lower()
        if len(query) < 2:
            return []
        starts: List[dict] = []
        contains: List[dict] = []
        for row in self._title_index:
            title_lower = row["title_lower"]
            if title_lower.startswith(query):
                starts.append(row)
            elif query in title_lower:
                contains.append(row)
            if len(starts) >= limit:
                break
        combined = starts if len(starts) >= limit else starts + contains
        return [
            {
                "title": row["title"],
                "poster": row["poster"],
                "release_year": row["release_year"],
                "genre": row["genre"],
                "rating": row["rating"],
            }
            for row in combined[:limit]
        ]

    def get_movie(self, title: str) -> dict:
        self._require_training()
        return self.movies.iloc[self.find_best_match_index(title)].to_dict()

    def filter_movies(self, genre="", region="", year="", language="", rating="") -> pd.DataFrame:
        """Filter movies for the home page catalog."""
        self._require_training()
        result = self.movies.copy()
        if genre:
            result = result[result["genre"].str.contains(genre, case=False, na=False)]
        if region:
            result = result[result["region"].str.lower() == region.lower()]
        if year:
            result = result[result["release_year"] == int(year)]
        if language:
            result = result[result["language"].str.lower() == language.lower()]
        if rating:
            result = result[result["rating"] >= float(rating)]
        return result.head(48)

    def options(self) -> dict:
        self._require_training()
        genres = sorted({g for text in self.movies["genre"] for g in str(text).split("|")})
        return {
            "genres": genres,
            "regions": sorted(self.movies["region"].dropna().unique()),
            "languages": sorted(self.movies["language"].dropna().unique()),
            "years": sorted(self.movies["release_year"].dropna().unique(), reverse=True)[:60],
        }

    def save(self, path: Path) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def _passes_filters(self, movie: dict, filters: Dict[str, str]) -> bool:
        return (
            (not filters.get("genre") or filters["genre"].lower() in str(movie["genre"]).lower()) and
            (not filters.get("region") or filters["region"].lower() == str(movie["region"]).lower()) and
            (not filters.get("language") or filters["language"].lower() == str(movie["language"]).lower()) and
            (not filters.get("year") or int(movie["release_year"]) == int(filters["year"])) and
            (not filters.get("rating") or float(movie["rating"]) >= float(filters["rating"]))
        )

    @staticmethod
    def _filter_alignment_boost(selected: pd.Series, movie: dict, filters: Dict[str, str]) -> float:
        """Prefer recommendations that align with the selected film when browsing without strict filters."""
        boost = 0.0
        if not any(str(filters.get(key, "")).strip() for key in ("genre", "region", "language", "year")):
            if str(movie.get("region", "")).lower() == str(selected.get("region", "")).lower():
                boost += 4.0
            if str(movie.get("language", "")).lower() == str(selected.get("language", "")).lower():
                boost += 2.0
            selected_genres = {g.strip().lower() for g in str(selected.get("genre", "")).split("|") if g.strip()}
            movie_genres = {g.strip().lower() for g in str(movie.get("genre", "")).split("|") if g.strip()}
            if selected_genres & movie_genres:
                boost += 3.0
            if int(movie.get("release_year", 0)) == int(selected.get("release_year", 0)):
                boost += 1.5
        return boost

    def _require_training(self) -> None:
        if not self.is_trained:
            self.train()
