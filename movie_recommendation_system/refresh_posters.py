"""Refresh movie poster URLs and real summaries from TMDB for the full catalog.

Set TMDB_API_KEY in your environment, then run:
    python refresh_posters.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from metadata_service import fix_catalog_metadata
from utils import DATA_DIR, fix_poster_url, normalize_title


TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def fetch_tmdb_movie(title: str, year: object, api_key: str) -> dict:
    """Return the best TMDB movie match for a title and year."""
    params = {
        "api_key": api_key,
        "query": title,
        "include_adult": "false",
    }
    if str(year).strip():
        params["year"] = str(year).strip()
    response = requests.get(TMDB_SEARCH_URL, params=params, timeout=12)
    response.raise_for_status()
    results = response.json().get("results", [])
    return results[0] if results else {}


def refresh_file(csv_path: Path, api_key: str) -> int:
    """Update poster URLs in one CSV file and return the number changed."""
    movies = fix_catalog_metadata(pd.read_csv(csv_path))
    updated_count = 0
    poster_cache: dict[str, str] = {}
    for row_index, row in movies.iterrows():
        title = str(row.get("title", "")).strip()
        year = row.get("release_year", "")
        cache_key = f"{normalize_title(title)}:{year}"
        if cache_key not in poster_cache:
            try:
                tmdb_movie = fetch_tmdb_movie(title, year, api_key)
                poster_path = tmdb_movie.get("poster_path")
                poster_cache[cache_key] = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else ""
                overview = str(tmdb_movie.get("overview", "")).strip()
                if overview:
                    movies.at[row_index, "synopsis"] = overview
                time.sleep(0.04)
            except requests.RequestException:
                poster_cache[cache_key] = ""
        new_poster = poster_cache[cache_key] or fix_poster_url(title, row.get("poster", ""), year)
        if new_poster and new_poster != row.get("poster", ""):
            movies.at[row_index, "poster"] = new_poster
            updated_count += 1
    movies.to_csv(csv_path, index=False)
    return updated_count


def main() -> None:
    """Refresh posters in both raw and cleaned movie CSV files."""
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set TMDB_API_KEY before running this script.")
    total_updated = 0
    for filename in ["movies.csv", "cleaned_movies.csv"]:
        csv_path = DATA_DIR / filename
        if csv_path.exists():
            total_updated += refresh_file(csv_path, api_key)
    print(f"Poster refresh complete. Updated {total_updated} poster URLs.")


if __name__ == "__main__":
    main()
