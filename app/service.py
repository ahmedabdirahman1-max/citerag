"""Orchestration: question -> retrieve -> guardrail -> grounded answer + citations.

Returns a transparent envelope: the answer, whether it was refused, and the
source chunks behind it (document, section, snippet, similarity score). Every
answer is auditable back to the text it came from.
"""
from __future__ import annotations
from typing import Any, Dict

from .config import TOP_K
from .generate import generate_answer
from .guardrail import REFUSAL_TEXT, should_refuse
from .index import TfidfIndex

SAMPLE_QUESTIONS = [
    "What is the return window for items?",
    "Is there a restocking fee?",
    "Can I return clearance items?",
    "When is shipping free?",
    "Do you ship to Hawaii or PO boxes?",
    "What does the warranty cover?",
    "What is the company's annual revenue?",  # intentionally out-of-scope -> refusal
]


def _snippet(text: str, limit: int = 180) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def answer_question(query: str, index: TfidfIndex) -> Dict[str, Any]:
    retrieved = index.query(query, TOP_K)
    top_score = retrieved[0][1] if retrieved else 0.0

    sources = [
        {
            "n": i + 1,
            "id": c.id,
            "doc": c.doc,
            "section": c.section,
            "snippet": _snippet(c.text),
            "score": round(score, 4),
        }
        for i, (c, score) in enumerate(retrieved)
        if score > 0
    ]

    if should_refuse(top_score):
        return {
            "query": query,
            "refused": True,
            "answer": REFUSAL_TEXT,
            "mode": "guardrail",
            "top_score": round(top_score, 4),
            "sources": sources,  # shown for transparency, even on refusal
        }

    answer, mode = generate_answer(query, retrieved)
    refused = answer.strip().lower().startswith("i couldn't find")

    return {
        "query": query,
        "refused": refused,
        "answer": answer,
        "mode": mode,
        "top_score": round(top_score, 4),
        "sources": sources,
    }
