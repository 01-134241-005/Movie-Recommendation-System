"""Utility helpers for the Movie Recommendation System."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List
from urllib.parse import quote_plus

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
POSTERS_DIR = BASE_DIR / "static" / "posters"

# Verified TMDB image paths for well-known movies in the bundled catalog.
# Unknown titles use a neutral title-card poster so the app never shows a wrong random image.
KNOWN_POSTERS = {
    "toy story": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
    "jumanji": "https://image.tmdb.org/t/p/w500/vgpXmVaVyUL7GGiDeiK1mKEKzcX.jpg",
    "red heat": "https://www.impawards.com/1988/posters/red_heat.jpg",
    "the heat": "https://www.impawards.com/2013/posters/heat.jpg",
    "white heat": "https://www.impawards.com/1949/posters/white_heat.jpg",
    "body heat": "https://www.impawards.com/1981/posters/body_heat.jpg",
    "city heat": "https://www.impawards.com/1984/posters/city_heat.jpg",
    "fire down below": "https://www.impawards.com/1997/posters/fire_down_below.jpg",
    "sudden death": "https://www.impawards.com/1995/posters/sudden_death.jpg",
    "dracula dead and loving it": "https://www.impawards.com/1995/posters/dracula_dead_and_loving_it.jpg",
    "the american president": "https://www.impawards.com/1995/posters/american_president.jpg",
    "heat": "https://image.tmdb.org/t/p/w500/umSVjVdbVwtx5ryCA2QXL44Durm.jpg",
    "sabrina": "https://image.tmdb.org/t/p/w500/z1oNjotUI7D06J4LWQFQzdIuPnf.jpg",
    "goldeneye": "https://image.tmdb.org/t/p/w500/z0ljRnNxIO7CRBhLEO0DvLgAFPR.jpg",
    "balto": "https://image.tmdb.org/t/p/w500/gV5PCAVCPNxlOLFM1bKk50EqLXO.jpg",
    "casino": "https://image.tmdb.org/t/p/w500/4TS5O1IP42bY2BvgMxL156EENy.jpg",
    "sense and sensibility": "https://image.tmdb.org/t/p/w500/kSCvfuI6Hfu2Uro7T6D0IW3vmkY.jpg",
    "four rooms": "https://image.tmdb.org/t/p/w500/75aHn1NOYXh4M7L5shoeQ6NGykP.jpg",
    "ace ventura when nature calls": "https://image.tmdb.org/t/p/w500/wRlGnJhEzcxBjvWtvbjhDSU1cIY.jpg",
    "the dark knight": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "batman begins": "https://image.tmdb.org/t/p/w500/4MpN4kIEqUjW8OPtOQJXlTdHiJV.jpg",
    "the batman": "https://image.tmdb.org/t/p/w500/74xTEgt7R36Fpooo50r9T25onhq.jpg",
    "inception": "https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
    "interstellar": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    "dangal": "https://image.tmdb.org/t/p/w500/cJRPOLEexI7qp2DKtFfCh7YaaUG.jpg",
    "3 idiots": "https://image.tmdb.org/t/p/w500/66A9MqXOyVFCssoloscw79z8Tew.jpg",
    "pathaan": "https://image.tmdb.org/t/p/w500/m1b9toKYyCujHuLoXB5GSDunO9e.jpg",
    "parasite": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
    "train to busan": "https://image.tmdb.org/t/p/w500/vNVFt6dtcqnI7hqa6LFBUibuFiw.jpg",
    "oldboy": "https://image.tmdb.org/t/p/w500/pWDtjs568ZfOTMbURQBYuT4Qxka.jpg",
    "spirited away": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
    "your name": "https://image.tmdb.org/t/p/w500/q719jXXEzOoYaps6babgKnONONX.jpg",
    "akira": "https://image.tmdb.org/t/p/w500/neZ0ykEsPqxamsX6o5QNUFILQrz.jpg",
    "amelie": "https://image.tmdb.org/t/p/w500/oTKduWL2tpIKEmkAqF4mFEAWAsv.jpg",
    "the lives of others": "https://image.tmdb.org/t/p/w500/5BCyeLJHPcRwhu0YaRqUzw00JJ4.jpg",
    "life is beautiful": "https://image.tmdb.org/t/p/w500/mfnkSeeVOBVheuyn2lo4tfmOPQb.jpg",
    "mean girls": "https://image.tmdb.org/t/p/w500/8EsmHvt46OPEw69pw9R5s4LA7TE.jpg",
    "clueless": "https://image.tmdb.org/t/p/w500/u0z5cq6IZWJvYDhhhJl6Y9z2Q1R.jpg",
}

KNOWN_SYNOPSES = {
    "toy story": "A cowboy doll feels threatened when a new space ranger toy becomes his owner's favorite.",
    "jumanji": "Two children release a magical board game's wild dangers into the real world.",
    "heat": "A master thief and a determined detective move toward a final confrontation in Los Angeles.",
    "sabrina": "A chauffeur's daughter returns from Paris and becomes caught between two wealthy brothers.",
    "goldeneye": "James Bond faces a former ally who plans to use a stolen satellite weapon.",
    "sudden death": "A firefighter must stop terrorists during a packed championship hockey game.",
    "the american president": "A widowed U.S. president risks his political future when he falls in love with a lobbyist.",
    "dracula dead and loving it": "Mel Brooks parodies the Dracula legend with a comic take on the classic vampire story.",
    "balto": "A brave sled dog helps deliver medicine during a dangerous Alaskan diphtheria outbreak.",
    "nixon": "Oliver Stone's drama examines Richard Nixon's rise, presidency, and downfall.",
    "cutthroat island": "A pirate captain hunts for treasure while outrunning enemies across the Caribbean.",
    "casino": "A Las Vegas casino operator is pulled into corruption, violence, and betrayal.",
    "sense and sensibility": "Two sisters face love, loss, and social pressure after their family fortune disappears.",
    "four rooms": "A hotel bellhop survives four strange encounters during one chaotic New Year's Eve.",
    "ace ventura when nature calls": "Ace Ventura travels to Africa to find a sacred missing animal and prevent conflict.",
    "money train": "Two foster-brother transit cops plan a risky robbery of a New York subway money train.",
    "get shorty": "A loan shark enters Hollywood and discovers movie production can be as ruthless as crime.",
    "copycat": "A criminal psychologist and detective hunt a serial killer copying infamous murderers.",
    "assassins": "Two rival hitmen collide while chasing the same high-value target.",
    "powder": "A sheltered young man with unusual powers struggles to find acceptance.",
    "leaving las vegas": "A self-destructive writer and a lonely sex worker form a fragile bond in Las Vegas.",
    "othello": "Shakespeare's tragedy follows jealousy, manipulation, and betrayal around a respected general.",
    "dawn of the dead": "Survivors of a zombie outbreak take shelter inside a shopping mall as society collapses.",
    "red heat": "A Soviet police officer teams with a Chicago detective to catch a drug dealer.",
    "the heat": "An FBI agent and a Boston detective clash while taking down a drug lord.",
    "white heat": "A volatile gangster plans one last job while police close in around him.",
    "body heat": "A lawyer begins a dangerous affair that leads him into murder and betrayal.",
    "city heat": "A private eye and a police lieutenant cross paths during a 1930s crime case.",
    "fire down below": "An undercover agent investigates toxic dumping in a small Appalachian town.",
}


def ensure_directories() -> None:
    """Create project folders used by the application."""
    for folder in [DATA_DIR, CHARTS_DIR, POSTERS_DIR, BASE_DIR / "screenshots"]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    """Return normalized text suitable for TF-IDF processing."""
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s,.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: object) -> str:
    """Normalize catalog titles so poster lookups are stable."""
    text = re.sub(r"\([^)]*\)", "", str(title)).strip()
    if ", the" in text.lower():
        text = "The " + re.sub(r",\s*The", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


def split_pipe_values(value: object) -> List[str]:
    """Split pipe-separated values used in the CSV dataset."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def is_placeholder_url(url: object) -> bool:
    """Return True when the URL is not a real film poster."""
    text = str(url or "").strip().lower()
    return not text or "placehold.co" in text or "picsum.photos" in text


def title_card_url(title: str, year: object = "") -> str:
    """Return a professional fallback image that cannot be mistaken for another film."""
    label = str(title).strip() or "Movie"
    if str(year).strip():
        label = f"{label}\n{year}"
    return "https://placehold.co/420x630/111827/e5e7eb/png?text=" + quote_plus(label)


def poster_url(title: str, index: int = 0, year: object = "") -> str:
    """Return the correct known poster or a neutral title-card fallback."""
    return KNOWN_POSTERS.get(normalize_title(title), title_card_url(title, year or index))


def fix_poster_url(title: object, current_url: object = "", year: object = "") -> str:
    """Replace random placeholder posters with reliable poster assets."""
    known_poster = KNOWN_POSTERS.get(normalize_title(title))
    if known_poster:
        return known_poster
    current = str(current_url or "").strip()
    if current and not is_placeholder_url(current):
        return current
    return title_card_url(str(title), year)


def fallback_synopsis(title: object, genre: object = "", year: object = "") -> str:
    """Create a short neutral synopsis when a real overview is unavailable."""
    title_text = str(title).strip() or "This film"
    year_text = f" from {year}" if str(year).strip() else ""
    genre_items = [
        "family" if item.strip().lower() == "children" else item.strip().lower()
        for item in str(genre).split("|")
        if item.strip()
    ]
    if genre_items:
        genre_text = ", ".join(genre_items[:3])
        article = "an" if genre_text[0] in "aeiou" else "a"
        return f"{title_text} is {article} {genre_text} film{year_text}."
    return f"{title_text} is a feature film{year_text}."


def fix_synopsis(title: object, current_synopsis: object = "", genre: object = "", year: object = "") -> str:
    """Return a clean movie-facing synopsis with no project/dataset wording."""
    known_synopsis = KNOWN_SYNOPSES.get(normalize_title(title))
    if known_synopsis:
        return known_synopsis
    text = str(current_synopsis or "").strip()
    banned_phrases = [
        "public movielens catalog",
        "metadata",
        "ai recommendation",
        "recommendation experiments",
        "movie catalog",
    ]
    if not text or any(phrase in text.lower() for phrase in banned_phrases):
        return fallback_synopsis(title, genre, year)
    return text


def percentage(value: float) -> str:
    """Format a similarity score as a percentage."""
    return f"{max(0.0, min(1.0, float(value))) * 100:.1f}%"


def unique_sorted(values: Iterable[object]) -> List[str]:
    """Return sorted unique non-empty strings from an iterable."""
    cleaned = {str(item).strip() for item in values if str(item).strip()}
    return sorted(cleaned)
