# app/utils/utils.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Returns the current UTC datetime object
def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# Returns the current UTC time formatted as an ISO 8601 string (e.g., 2026-04-11T12:00:00+00:00)
def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return utc_now().isoformat()


# Generates a unique run ID based on current UTC time, formatted for filenames
def create_run_id() -> str:
    """Create a filesystem-friendly UTC run id."""
    return utc_now().strftime("%Y%m%d_%H%M%S")


# Ensures that a directory exists (creates it if necessary) and returns it as a Path object
def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Saves Python data as a JSON file with UTF-8 encoding and readable formatting
def save_json(data: Any, path: str | Path) -> None:
    """Save JSON using UTF-8 and pretty formatting."""
    path = Path(path)
    ensure_dir(path.parent)  # Ensure the directory exists before saving
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Loads and returns JSON data from a file
def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Saves plain text to a file (overwrites if file already exists)
def save_text(text: str, path: str | Path) -> None:
    """Save plain text to disk."""
    path = Path(path)
    ensure_dir(path.parent)  # Ensure the directory exists before saving
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# Appends a single line of text to a file (creates file if it does not exist)
def append_text_line(text: str, path: str | Path) -> None:
    """Append one line to a text file."""
    path = Path(path)
    ensure_dir(path.parent)  # Ensure the directory exists before appending
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")  # Remove trailing spaces and add newline


# Checks whether a given file or directory path exists
def path_exists(path: str | Path) -> bool:
    """Return True if the path exists."""
    return Path(path).exists()