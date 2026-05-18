"""Load API keys from a project .env file into os.environ."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_LOADED = False
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Read .env once so TMDB_API_KEY and OMDB_API_KEY are available."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if not ENV_PATH.exists():
        _ENV_LOADED = True
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)
    _ENV_LOADED = True


def tmdb_api_configured() -> bool:
    load_env()
    return bool(os.getenv("TMDB_API_KEY", "").strip())
