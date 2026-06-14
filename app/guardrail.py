"""The anti-hallucination guardrail.

If the best retrieved chunk isn't similar enough to the question, the honest
answer is "I don't know" — not a confident guess assembled from loosely related
text. Refusing on weak retrieval is the single most important behaviour here:
it's what stops the system from inventing answers to out-of-scope questions.
"""
from __future__ import annotations

from .config import REFUSAL_THRESHOLD

REFUSAL_TEXT = "I couldn't find that in the documents."


def should_refuse(top_score: float, threshold: float = REFUSAL_THRESHOLD) -> bool:
    return top_score < threshold
