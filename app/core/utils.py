# app/core/utils.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return utc_now().isoformat()


def create_run_id() -> str:
    """Create a filesystem-friendly UTC run id."""
    return utc_now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON using UTF-8 and pretty formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str | Path) -> None:
    """Save plain text to disk."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def append_text_line(text: str, path: str | Path) -> None:
    """Append one line to a text file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def path_exists(path: str | Path) -> bool:
    """Return True if the path exists."""
    return Path(path).exists()