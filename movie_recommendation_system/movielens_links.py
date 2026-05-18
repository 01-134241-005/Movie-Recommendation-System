"""Load MovieLens movieId -> tmdbId/imdbId links for poster resolution."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from utils import DATA_DIR

LINKS_PATH = DATA_DIR / "links.csv"
MOVIELENS_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def ensure_links_file() -> Path:
    """Download and cache MovieLens links.csv when missing."""
    if LINKS_PATH.exists():
        return LINKS_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = urllib.request.urlopen(MOVIELENS_ZIP_URL, timeout=90).read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        links = pd.read_csv(bundle.open("ml-latest-small/links.csv"))
    links.to_csv(LINKS_PATH, index=False)
    return LINKS_PATH


def load_links() -> pd.DataFrame:
    ensure_links_file()
    return pd.read_csv(LINKS_PATH)
