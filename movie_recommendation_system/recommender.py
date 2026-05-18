"""High-level service used by Flask routes."""

from __future__ import annotations

from functools import lru_cache

from model import KNNRecommender
from poster_service import PosterResolver
from utils import is_placeholder_url


class MovieRecommendationService:
    """Small service layer between UI and the ML model."""

    def __init__(self):
        self.posters = PosterResolver()
        self.model = KNNRecommender()
        self.model.train()
        self._apply_cached_posters()
        self.featured_movies = [
            {
                "title": "The Dark Knight",
                "poster": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
                "backdrop": "https://image.tmdb.org/t/p/original/hkBaDkMWbLaf8B1lsWsKX7Ew3Xq.jpg",
                "trailer": "EXeTwQWrcwY",
                "genre": "Action / Crime / Drama",
                "rating": 9.0,
                "year": 2008,
            },
            {
                "title": "Inception",
                "poster": "https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
                "backdrop": "https://image.tmdb.org/t/p/original/s3TBrRGB1iav7gFOCNx3H31MoES.jpg",
                "trailer": "YoHD9XEInc0",
                "genre": "Action / Sci-Fi / Thriller",
                "rating": 8.8,
                "year": 2010,
            },
            {
                "title": "Interstellar",
                "poster": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
                "backdrop": "https://image.tmdb.org/t/p/original/rAiYTfKGqDCRIIqo664sY9XZIvQ.jpg",
                "trailer": "zSWdZVtXT7E",
                "genre": "Adventure / Drama / Sci-Fi",
                "rating": 8.7,
                "year": 2014,
            },
            {
                "title": "Parasite",
                "poster": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
                "backdrop": "https://image.tmdb.org/t/p/original/TU9NIjwzjoKPwQHoHshkFcQUCG.jpg",
                "trailer": "5xH0HfJHsaY",
                "genre": "Drama / Thriller",
                "rating": 8.5,
                "year": 2019,
            },
            {
                "title": "Spirited Away",
                "poster": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
                "backdrop": "https://image.tmdb.org/t/p/original/mSDsSDwaP3E7dEfUPWy4J0djt4O.jpg",
                "trailer": "ByXuk9QqQkk",
                "genre": "Animation / Fantasy",
                "rating": 8.6,
                "year": 2001,
            },
            {
                "title": "Toy Story",
                "poster": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
                "backdrop": "https://image.tmdb.org/t/p/original/3Rfvhy1Nl6sSGJwyjb0QiZzZYlB.jpg",
                "trailer": "KYz2wyBy3kc",
                "genre": "Animation / Comedy / Family",
                "rating": 8.3,
                "year": 1995,
            },
            {
                "title": "Batman Begins",
                "poster": "https://image.tmdb.org/t/p/w500/4MpN4kIEqUjW8OPtOQJXlTdHiJV.jpg",
                "backdrop": "https://image.tmdb.org/t/p/original/ew5FcYiRhTYNJAkxoVPMNlCOdVn.jpg",
                "trailer": "neY2xVmOfUM",
                "genre": "Action / Adventure",
                "rating": 8.2,
                "year": 2005,
            },
        ]

    def home_movies(self, filters: dict):
        movies = self.model.filter_movies(**filters).to_dict("records")
        return self.posters.attach_catalog_posters(movies)

    def recommendations_for(self, title: str, filters: dict, fast: bool = False):
        selected, recommendations = self._recommendations_cached(title, self._filters_key(filters))
        if fast:
            return selected, recommendations
        return (
            self.posters.attach_catalog_posters([selected])[0],
            self.posters.attach_catalog_posters(recommendations),
        )

    def suggestions(self, query: str):
        return self.posters.attach_catalog_posters(self.model.search_titles(query))

    def poster_for(self, title: str, year: object = "", current_url: object = "") -> str:
        return self.posters.catalog_poster(title, year, current_url)

    def filter_options(self):
        return self.model.options()

    def featured(self):
        return self.featured_movies

    @lru_cache(maxsize=512)
    def _recommendations_cached(self, title: str, filters_key: tuple) -> tuple:
        filters = dict(filters_key)
        selected = self.model.get_movie(title)
        recommendations = self.model.recommend(title, top_n=10, filters=filters)
        return selected, tuple(recommendations)

    @staticmethod
    def _filters_key(filters: dict) -> tuple:
        return tuple(sorted((key, str(value)) for key, value in (filters or {}).items()))

    def _apply_cached_posters(self) -> None:
        """Apply cached real poster URLs to the in-memory catalog."""
        for index, row in self.model.movies.iterrows():
            self.model.movies.at[index, "poster"] = self.posters.catalog_poster(
                row["title"],
                row["release_year"],
                row.get("poster", ""),
                row.get("movie_id", ""),
            )
        if hasattr(self.model, "_title_index"):
            for entry in self.model._title_index:
                entry["poster"] = self.posters.catalog_poster(
                    entry["title"],
                    entry["release_year"],
                    entry.get("poster", ""),
                )
