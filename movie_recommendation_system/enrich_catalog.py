"""Enrich movies.csv with real TMDB posters for the full catalog.

Uses MovieLens tmdbId links + TMDB API (if TMDB_API_KEY is set) or TMDB public pages.
Run with Spyder/main.py stopped so movies.csv is not locked:

    python enrich_catalog.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from env_config import ENV_PATH, load_env
from metadata_service import fix_catalog_metadata
from movielens_links import ensure_links_file, load_links
from poster_service import PosterResolver
from utils import DATA_DIR, is_placeholder_url

BATCH_SIZE = 100
WORKERS = 4
REQUEST_PAUSE = 0.22


def _safe_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write CSV via a temp file so Windows file locks do not abort long runs."""
    temp_path = path.with_suffix(".tmp.csv")
    for attempt in range(5):
        try:
            frame.to_csv(temp_path, index=False)
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1.5)


def _has_real_poster(url: object) -> bool:
    text = str(url or "").strip().lower()
    if is_placeholder_url(text):
        return False
    if "upload.wikimedia.org" in text:
        return False
    return "image.tmdb.org" in text or "media.themoviedb.org" in text


def _resolve_row(resolver: PosterResolver, index: int, row: pd.Series) -> tuple[int, str]:
    time.sleep(REQUEST_PAUSE)
    poster = resolver._fetch_poster(
        row.get("title", ""),
        row.get("release_year", ""),
        movie_id=row.get("movie_id", ""),
        tmdb_id=row.get("tmdbId", ""),
    )
    return index, poster


def enrich_dataframe(movies: pd.DataFrame, resolver: PosterResolver) -> pd.DataFrame:
    movies = fix_catalog_metadata(movies)
    links = load_links()
    movies = movies.merge(links[["movieId", "tmdbId"]], left_on="movie_id", right_on="movieId", how="left")
    if "movieId" in movies.columns:
        movies = movies.drop(columns=["movieId"])

    pending = []
    for index, row in movies.iterrows():
        if not _has_real_poster(row.get("poster", "")):
            pending.append((index, row))

    total = len(pending)
    updated = 0
    started = time.time()
    mode = "TMDB API" if resolver.api_key else "TMDB public pages"
    print(f"Resolving posters for {total} titles via {mode} ({WORKERS} workers)...", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = pending[batch_start: batch_start + BATCH_SIZE]
            futures = [pool.submit(_resolve_row, resolver, index, row) for index, row in batch]
            for future in as_completed(futures):
                index, poster = future.result()
                if _has_real_poster(poster):
                    movies.at[index, "poster"] = poster
                    updated += 1
            resolver.save_cache()
            save_frame = movies.drop(columns=["tmdbId"], errors="ignore")
            _safe_write_csv(save_frame, DATA_DIR / "movies.csv")
            done = min(batch_start + len(batch), total)
            elapsed = max(time.time() - started, 1)
            rate = done / elapsed
            remaining = (total - done) / rate if rate else 0
            print(
                f"  {done}/{total} | {updated} posters | "
                f"{rate:.1f}/s | ~{remaining / 60:.0f} min left",
                flush=True,
            )

    if "tmdbId" in movies.columns:
        movies = movies.drop(columns=["tmdbId"])
    resolver.save_cache()
    print(f"Poster enrichment finished. Updated {updated} rows.")
    return movies


def main() -> None:
    load_env()
    ensure_links_file()
    csv_path = DATA_DIR / "movies.csv"
    movies = pd.read_csv(csv_path)
    print(f"Loaded {len(movies)} movies from {csv_path}")
    before = movies["poster"].astype(str).apply(lambda url: not _has_real_poster(url)).sum()
    print(f"Missing real posters before enrichment: {before}")

    resolver = PosterResolver()
    if resolver.api_key:
        print("Using TMDB_API_KEY for fast poster lookup.")
    else:
        print(f"No TMDB_API_KEY found. Create {ENV_PATH} with TMDB_API_KEY=your_key")
        print("Falling back to public TMDB pages (~3x slower).")

    resolver.cache = {
        key: url for key, url in resolver.cache.items() if _has_real_poster(url)
    }
    resolver._dirty = bool(resolver.cache)

    movies = enrich_dataframe(movies, resolver)
    _safe_write_csv(movies, csv_path)
    cleaned_path = DATA_DIR / "cleaned_movies.csv"
    if cleaned_path.exists():
        cleaned_path.unlink()
    after = movies["poster"].astype(str).apply(lambda url: not _has_real_poster(url)).sum()
    print(f"Missing real posters after enrichment: {after}")


if __name__ == "__main__":
    main()
