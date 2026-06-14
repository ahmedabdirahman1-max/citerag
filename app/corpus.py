"""Load markdown documents and split them into citable chunks.

Each document is split by its `## ` section headings, so a chunk is a coherent
section (e.g. "Restocking fee: ..."). That granularity makes citations meaningful
— an answer points to a specific section, not a whole document.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import DOCS_DIR


@dataclass
class Chunk:
    id: str          # e.g. "returns_policy#2"
    doc: str         # human title, e.g. "Returns & Refund Policy"
    section: str     # heading of this chunk, e.g. "Restocking fee"
    text: str        # full chunk text (heading + body)


def _title_from(lines: List[str], fallback: str) -> str:
    for line in lines:
        if line.startswith("# "):
            return line[2:].split("—")[-1].strip() or fallback
    return fallback


def _split_sections(body: str):
    """Yield (heading, text) pairs split on '## ' headings."""
    parts = re.split(r"(?m)^##\s+", body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        heading = lines[0].strip()
        yield heading, part


def load_chunks(docs_dir: Path | None = None) -> List[Chunk]:
    docs_dir = docs_dir or DOCS_DIR
    chunks: List[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        title = _title_from(lines, path.stem.replace("_", " ").title())
        # Drop the top-level "# Title" line before sectioning.
        body = re.sub(r"(?m)^#\s+.*$", "", raw, count=1)
        for i, (heading, text) in enumerate(_split_sections(body)):
            chunks.append(Chunk(
                id=f"{path.stem}#{i}",
                doc=title,
                section=heading,
                text=text,
            ))
    return chunks
