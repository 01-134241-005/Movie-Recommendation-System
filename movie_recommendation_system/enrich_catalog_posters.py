"""Bulk-fetch authentic posters and save them into the movie CSV files."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from poster_service import PosterResolver
from utils import DATA_DIR, is_placeholder_url


def fetch_one(resolver: PosterResolver, title: str, year: object, current: str) -> tuple[str, str]:
    return title, resolver.resolve_online(title, year, current)


def enrich_file(csv_path, resolver: PosterResolver, workers: int = 10) -> int:
    movies = pd.read_csv(csv_path)
    tasks = [
        (str(row["title"]), row["release_year"], str(row.get("poster", "")))
        for _, row in movies.iterrows()
        if str(row.get("title", "")).strip() and is_placeholder_url(row.get("poster", ""))
    ]
    updated = 0
    print(f"{csv_path.name}: fetching {len(tasks)} posters...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, resolver, *task) for task in tasks]
        done = 0
        for future in as_completed(futures):
            title, poster = future.result()
            if not poster or is_placeholder_url(poster):
                done += 1
                continue
            mask = movies["title"] == title
            movies.loc[mask, "poster"] = poster
            updated += int(mask.sum())
            done += 1
            if done % 200 == 0:
                resolver.save_cache()
                movies.to_csv(csv_path, index=False)
                print(f"  {done}/{len(tasks)} processed, {updated} real posters")
    movies.to_csv(csv_path, index=False)
    resolver.save_cache()
    return updated


def main() -> None:
    resolver = PosterResolver()
    total = 0
    for name in ["movies.csv", "cleaned_movies.csv"]:
        path = DATA_DIR / name
        if path.exists():
            total += enrich_file(path, resolver)
    print(f"Done. Saved {total} authentic poster URLs.")


if __name__ == "__main__":
    main()
