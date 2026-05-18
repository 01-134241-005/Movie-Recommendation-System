"""Resolve real movie posters from TMDB and Wikipedia with persistent caching."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from env_config import load_env
from utils import DATA_DIR, KNOWN_POSTERS, is_placeholder_url, normalize_title, title_card_url

load_env()

_TMDB_ID_LOOKUP: Optional[Dict[int, int]] = None

CACHE_PATH = DATA_DIR / "poster_cache.json"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
WIKI_API = "https://en.wikipedia.org/w/api.php"
OMDB_URL = "https://www.omdbapi.com/"


def display_title(title: object) -> str:
    """Convert MovieLens-style titles to readable search text."""
    text = str(title).strip()
    if ", the" in text.lower():
        text = "The " + re.sub(r",\s*The", "", text, flags=re.IGNORECASE).strip()
    return text


class PosterResolver:
    """Fetch and cache original film posters for catalog titles."""

    def __init__(self):
        self.api_key = os.getenv("TMDB_API_KEY", "").strip()
        self.omdb_key = os.getenv("OMDB_API_KEY", "").strip()
        self.cache: Dict[str, str] = self._load_cache()
        self._dirty = False
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "CinephileWorld/1.0"})

    def catalog_poster(
        self,
        title: object,
        year: object = "",
        current_url: object = "",
        movie_id: object = "",
    ) -> str:
        """Fast lookup: known map, CSV URL, or cache only (no network)."""
        known = KNOWN_POSTERS.get(normalize_title(title))
        if known:
            return known
        current = str(current_url or "").strip()
        if current and not is_placeholder_url(current):
            return current
        cached = self.cache.get(self._cache_key(title, year), "")
        if cached and not is_placeholder_url(cached):
            return cached
        return current or title_card_url(str(title), year)

    def attach_catalog_posters(self, records: Iterable[dict]) -> List[dict]:
        """Attach poster URLs from catalog/cache without network calls."""
        enriched = []
        for record in records:
            movie = dict(record)
            movie["poster"] = self.catalog_poster(
                movie.get("title", ""),
                movie.get("release_year", ""),
                movie.get("poster", ""),
                movie.get("movie_id", ""),
            )
            enriched.append(movie)
        return enriched

    def resolve_online(
        self,
        title: object,
        year: object = "",
        current_url: object = "",
        movie_id: object = "",
        tmdb_id: object = "",
    ) -> str:
        """Resolve with network (for background poster jobs and /api/poster)."""
        poster = self.catalog_poster(title, year, current_url)
        if not is_placeholder_url(poster):
            return poster
        cache_key = self._cache_key(title, year)
        fetched = self._fetch_poster(title, year, movie_id=movie_id, tmdb_id=tmdb_id)
        if fetched and not is_placeholder_url(fetched):
            self.cache[cache_key] = fetched
            self._dirty = True
            return fetched
        return poster

    def save_cache(self) -> None:
        if self._dirty:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
            self._dirty = False

    def _fetch_poster(self, title: object, year: object, movie_id: object = "", tmdb_id: object = "") -> str:
        numeric_id = self._coerce_tmdb_id(tmdb_id) or self._tmdb_id_for(movie_id)
        if numeric_id:
            if self.api_key:
                poster = self._fetch_tmdb_by_id_api(numeric_id)
                if poster:
                    return poster
            poster = self._fetch_tmdb_page(numeric_id)
            if poster:
                return poster
        for fetcher in (self._fetch_tmdb, self._fetch_omdb, self._fetch_wikipedia):
            poster = fetcher(title, year)
            if poster:
                return poster
        return ""

    def _fetch_tmdb_by_id_api(self, tmdb_id: object) -> str:
        """Fetch poster_path from TMDB movie details (fastest when API key is set)."""
        numeric_id = self._coerce_tmdb_id(tmdb_id)
        if not numeric_id or not self.api_key:
            return ""
        try:
            response = self._session.get(
                f"https://api.themoviedb.org/3/movie/{numeric_id}",
                params={"api_key": self.api_key},
                timeout=8,
            )
            response.raise_for_status()
            poster_path = response.json().get("poster_path")
            return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else ""
        except requests.RequestException:
            return ""

    @staticmethod
    def _coerce_tmdb_id(value: object) -> int:
        try:
            if value is None or (isinstance(value, float) and value != value):
                return 0
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _fetch_tmdb_page(self, tmdb_id: object) -> str:
        """Read the official poster from a TMDB movie page (no API key required)."""
        numeric_id = self._coerce_tmdb_id(tmdb_id)
        if not numeric_id:
            return ""
        url = f"https://www.themoviedb.org/movie/{numeric_id}"
        for attempt in range(4):
            try:
                response = self._session.get(url, timeout=12)
                if response.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                match = re.search(r'property="og:image" content="([^"]+)"', response.text)
                if not match:
                    return ""
                poster = match.group(1).strip()
                return poster.replace("https://media.themoviedb.org", "https://image.tmdb.org")
            except requests.RequestException:
                if attempt == 3:
                    return ""
                time.sleep(0.8 * (attempt + 1))
        return ""

    def _tmdb_id_for(self, movie_id: object) -> int:
        lookup = self._tmdb_lookup()
        try:
            return int(lookup.get(int(float(movie_id)), 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _tmdb_lookup(cls) -> Dict[int, int]:
        global _TMDB_ID_LOOKUP
        if _TMDB_ID_LOOKUP is not None:
            return _TMDB_ID_LOOKUP
        links_path = DATA_DIR / "links.csv"
        if not links_path.exists():
            try:
                from movielens_links import ensure_links_file

                ensure_links_file()
            except Exception:
                _TMDB_ID_LOOKUP = {}
                return _TMDB_ID_LOOKUP
        try:
            import pandas as pd

            links = pd.read_csv(links_path)
            _TMDB_ID_LOOKUP = {
                int(row.movieId): int(row.tmdbId)
                for row in links.itertuples()
                if pd.notna(row.tmdbId)
            }
        except Exception:
            _TMDB_ID_LOOKUP = {}
        return _TMDB_ID_LOOKUP

    def _fetch_tmdb(self, title: object, year: object) -> str:
        if not self.api_key:
            return ""
        params = {
            "api_key": self.api_key,
            "query": display_title(title),
            "include_adult": "false",
        }
        if str(year).strip():
            params["year"] = str(year).strip()
        try:
            response = self._session.get(TMDB_SEARCH_URL, params=params, timeout=6)
            response.raise_for_status()
            results = response.json().get("results", [])
            if not results:
                return ""
            target_year = str(year).strip()
            best = results[0]
            if target_year:
                for item in results:
                    release = str(item.get("release_date", ""))[:4]
                    if release == target_year and item.get("poster_path"):
                        best = item
                        break
            poster_path = best.get("poster_path")
            return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else ""
        except requests.RequestException:
            return ""

    def _fetch_omdb(self, title: object, year: object) -> str:
        if not self.omdb_key:
            return ""
        params = {"apikey": self.omdb_key, "t": display_title(title)}
        if str(year).strip():
            params["y"] = str(year).strip()
        try:
            response = self._session.get(OMDB_URL, params=params, timeout=6)
            response.raise_for_status()
            poster = str(response.json().get("Poster", "")).strip()
            if poster and poster.lower() != "n/a":
                return poster
        except requests.RequestException:
            return ""
        return ""

    def _fetch_wikipedia(self, title: object, year: object) -> str:
        clean = display_title(title)
        queries = [
            f"{clean} ({year} film)" if str(year).strip() else "",
            f"{clean} {year} film" if str(year).strip() else "",
            f"{clean} film",
            clean,
        ]
        for search_query in queries:
            if not search_query:
                continue
            try:
                search_response = self._session.get(
                    WIKI_API,
                    params={
                        "action": "query",
                        "format": "json",
                        "list": "search",
                        "srsearch": search_query,
                        "srlimit": 1,
                    },
                    timeout=5,
                )
                search_response.raise_for_status()
                results = search_response.json().get("query", {}).get("search", [])
                if not results:
                    continue
                page_title = results[0]["title"]
                image_response = self._session.get(
                    WIKI_API,
                    params={
                        "action": "query",
                        "format": "json",
                        "titles": page_title,
                        "prop": "pageimages",
                        "pithumbsize": 500,
                    },
                    timeout=5,
                )
                image_response.raise_for_status()
                pages = image_response.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    source = str(page.get("thumbnail", {}).get("source", "")).strip()
                    if source:
                        return source
            except requests.RequestException:
                continue
        return ""

    @staticmethod
    def _cache_key(title: object, year: object) -> str:
        return f"{normalize_title(title)}:{str(year).strip()}"

    @staticmethod
    def _load_cache() -> Dict[str, str]:
        if not CACHE_PATH.exists():
            return {}
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            return {
                key: value
                for key, value in data.items()
                if value and not is_placeholder_url(value)
            }
        except (json.JSONDecodeError, OSError):
            return {}
