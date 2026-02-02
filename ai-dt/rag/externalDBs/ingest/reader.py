"""Read snippets from CSV, JSON, and plain text files."""

import csv
import json
import uuid
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    """Read CSV file. Expects columns: id (optional), text (required)."""
    snippets = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if not text:
                continue
            snippets.append({
                "id": row.get("id") or f"{path.stem}_{len(snippets)}",
                "text": text,
            })
    return snippets


def _read_json(path: Path) -> list[dict]:
    """Read JSON file. Expects list of {id?, text} or single {id?, text}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    snippets = []
    for item in data:
        text = item.get("text", "").strip()
        if not text:
            continue
        snippets.append({
            "id": item.get("id") or str(uuid.uuid4())[:8],
            "text": text,
        })
    return snippets


def _read_text(path: Path) -> list[dict]:
    """Read plain text file. Each non-empty paragraph (double newline separated) becomes a snippet."""
    content = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return [
        {"id": f"{path.stem}_{i}", "text": p}
        for i, p in enumerate(paragraphs)
    ]


_READERS = {
    ".csv": _read_csv,
    ".json": _read_json,
    ".txt": _read_text,
}


def read_snippets(input_path: str, fmt: str | None = None) -> list[dict]:
    """Read snippets from a file or directory.

    Args:
        input_path: Path to a file or directory.
        fmt: Force format ("csv", "json", "text"). Auto-detected if None.

    Returns:
        List of {"id": str, "text": str} dicts.
    """
    path = Path(input_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.iterdir())
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")

    snippets = []
    for f in files:
        if not f.is_file():
            continue
        ext = f".{fmt}" if fmt else f.suffix.lower()
        # Normalize format aliases
        if ext == ".text":
            ext = ".txt"
        reader = _READERS.get(ext)
        if reader is None:
            continue
        snippets.extend(reader(f))

    print(f"Read {len(snippets)} snippets from {input_path}")
    return snippets
