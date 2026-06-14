"""Runtime configuration, read from environment with safe defaults."""
from __future__ import annotations
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.environ.get("CITERAG_DOCS", str(BASE_DIR / "docs")))

# If set, answers are phrased by Claude *from the retrieved chunks only*. If empty,
# an extractive answer (the most relevant sentences, verbatim) is returned instead,
# so the app runs with no API key and no network.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-haiku-4-5-20251001").strip()

# How many chunks to retrieve, and the minimum similarity below which we refuse
# rather than answer. Refusing on weak retrieval is what prevents hallucination
# on out-of-scope questions.
TOP_K = int(os.environ.get("TOP_K", "4"))
REFUSAL_THRESHOLD = float(os.environ.get("REFUSAL_THRESHOLD", "0.06"))
