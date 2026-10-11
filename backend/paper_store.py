"""
Paper store: paper_id -> extracted text + processing state.

In-memory dict for speed, with a small on-disk cache (data/papers/<id>.json)
so extracted papers survive a server restart.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"


@dataclass
class Paper:
    paper_id: str
    filename: str
    page_count: int = 0
    chunk_count: int = 0
    chunks_done: int = 0
    # queued | processing | ready | failed
    state: str = "queued"
    stage: str = "Queued"
    text: str = ""
    warnings: list[str] = field(default_factory=list)
    error: dict[str, str] | None = None
    keyterms: list[str] = field(default_factory=list)
    summary: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def public(self, include_preview: bool = True) -> dict[str, Any]:
        out = {
            "paper_id": self.paper_id,
            "filename": self.filename,
            "state": self.state,
            "stage": self.stage,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "chunks_done": self.chunks_done,
            "char_count": len(self.text),
            "warnings": self.warnings,
            "error": self.error,
            "has_summary": self.summary is not None,
        }
        if include_preview and self.state == "ready":
            out["preview"] = self.text[:1200]
        return out


_lock = threading.Lock()
_papers: dict[str, Paper] = {}


def create(filename: str) -> Paper:
    paper = Paper(paper_id=uuid.uuid4().hex[:12], filename=filename)
    with _lock:
        _papers[paper.paper_id] = paper
    return paper


def get(paper_id: str) -> Paper | None:
    with _lock:
        paper = _papers.get(paper_id)
    if paper is None:
        paper = _load(paper_id)
        if paper:
            with _lock:
                _papers[paper_id] = paper
    return paper


def save(paper: Paper) -> None:
    """Persist a ready paper to disk (history is session-only, not persisted)."""
    if paper.state != "ready":
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(paper)
    data["history"] = []
    (DATA_DIR / f"{paper.paper_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _load(paper_id: str) -> Paper | None:
    if not paper_id.isalnum():
        return None
    path = DATA_DIR / f"{paper_id}.json"
    if not path.exists():
        return None
    try:
        return Paper(**json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError):
        return None
