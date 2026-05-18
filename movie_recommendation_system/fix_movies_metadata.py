"""Apply metadata corrections to movies.csv without network calls."""

from __future__ import annotations

import pandas as pd

from metadata_service import fix_catalog_metadata
from utils import DATA_DIR, fix_poster_url

if __name__ == "__main__":
    path = DATA_DIR / "movies.csv"
    movies = fix_catalog_metadata(pd.read_csv(path))
    movies["poster"] = movies.apply(
        lambda row: fix_poster_url(row["title"], row["poster"], row["release_year"]),
        axis=1,
    )
    movies.to_csv(path, index=False)
    cleaned = DATA_DIR / "cleaned_movies.csv"
    if cleaned.exists():
        cleaned.unlink()
    print(f"Updated metadata for {len(movies)} movies in {path}")
