"""Tokenisation with light normalisation.

Kept deliberately dependency-free (no nltk/torch) so the demo installs and runs
instantly. A crude suffix stemmer collapses return/returns/returned -> "return"
so a question and the document phrase it differently still match.
"""
from __future__ import annotations
import re
from typing import List

_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "to", "of", "for", "and", "or", "in", "on", "at",
    "it", "its", "i", "you", "we", "they", "he", "she", "my", "your", "our",
    "can", "could", "would", "should", "will", "if", "then", "than", "that",
    "this", "these", "those", "with", "as", "by", "from", "me", "us", "any",
    "how", "what", "when", "where", "why", "which", "who", "whom", "about",
    "there", "here", "have", "has", "had", "get", "got", "into", "out",
}


def _stem(token: str) -> str:
    """Very small suffix stripper — not linguistically correct, just consistent."""
    for suf in ("ing", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            if suf == "s" and token.endswith("ss"):
                break
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> List[str]:
    out = []
    for raw in _WORD.findall(text.lower()):
        if raw in _STOP or len(raw) == 1:
            continue
        out.append(_stem(raw))
    return out
