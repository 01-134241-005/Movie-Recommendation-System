"""Correct region, language, and cast metadata for the MovieLens-style catalog."""

from __future__ import annotations

import re
from typing import Dict, Tuple

import pandas as pd

from utils import normalize_title

SYNTHETIC_CAST_MARKERS = (
    "Arjun Kapoor",
    "Priya Sharma",
    "Neha Khan",
    "Kim Min-jun",
    "Park Seo-yeon",
    "Lee Ji-ho",
    "Haruto Sato",
    "Yui Tanaka",
    "Ren Kobayashi",
    "Lucas Moreau",
    "Sofia Rossi",
    "Anna Weber",
    "Robert Stone",
    "Emma Carter",
    "Michael Reed",
)

# Verified region and language for well-known titles in the catalog.
TITLE_METADATA: Dict[str, Tuple[str, str]] = {
    "dangal": ("Bollywood", "Hindi"),
    "3 idiots": ("Bollywood", "Hindi"),
    "pathaan": ("Bollywood", "Hindi"),
    "parasite": ("Korean", "Korean"),
    "train to busan": ("Korean", "Korean"),
    "oldboy": ("Korean", "Korean"),
    "spirited away": ("Anime", "Japanese"),
    "your name": ("Anime", "Japanese"),
    "akira": ("Anime", "Japanese"),
    "amelie": ("European", "French"),
    "the lives of others": ("European", "German"),
    "life is beautiful": ("European", "Italian"),
}

NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")


def is_synthetic_cast(cast: object) -> bool:
    text = str(cast or "")
    return any(marker in text for marker in SYNTHETIC_CAST_MARKERS)


def infer_region_language(title: object, genre: object = "") -> Tuple[str, str]:
    """Infer catalog region and language; MovieLens titles default to Hollywood."""
    key = normalize_title(title)
    if key in TITLE_METADATA:
        return TITLE_METADATA[key]
    text = str(title or "")
    if NON_LATIN_RE.search(text):
        if any(token in text.lower() for token in ("bollywood", "hindi")):
            return "Bollywood", "Hindi"
        if any(token in text.lower() for token in ("korean", "hangul")):
            return "Korean", "Korean"
        return "Asian", "Other"
    return "Hollywood", "English"


def fix_catalog_metadata(movies: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy cyclic region/language/cast assignments with sensible values."""
    fixed = movies.copy()
    for index, row in fixed.iterrows():
        region, language = infer_region_language(row.get("title", ""), row.get("genre", ""))
        fixed.at[index, "region"] = region
        fixed.at[index, "language"] = language
        if is_synthetic_cast(row.get("cast", "")):
            fixed.at[index, "cast"] = ""
    return fixed
