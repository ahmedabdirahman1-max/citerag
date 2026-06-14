"""Turn retrieved chunks into a grounded answer.

Two modes, same guarantee — the answer may only draw on the retrieved chunks:
  * extractive (default): returns the most relevant sentences verbatim, cited.
    No API key, no network, no chance of fabrication.
  * llm (optional): Claude phrases an answer FROM the supplied sources only, is
    told to cite inline and to refuse if the sources don't contain the answer.
"""
from __future__ import annotations
import re
from typing import List, Tuple

from .config import ANTHROPIC_API_KEY, ANSWER_MODEL
from .corpus import Chunk
from .guardrail import REFUSAL_TEXT

Retrieved = List[Tuple[Chunk, float]]


def _body(chunk: Chunk) -> str:
    """Chunk text without its heading line."""
    return "\n".join(chunk.text.splitlines()[1:]).strip() or chunk.text.strip()


def _first_sentences(text: str, n: int = 2) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(s for s in sents[:n] if s).strip()


def extractive_answer(retrieved: Retrieved) -> str:
    """Most relevant sentences, verbatim, cited to the top source."""
    top_chunk = retrieved[0][0]
    return f"{_first_sentences(_body(top_chunk), 2)} [1]"


def llm_answer(query: str, retrieved: Retrieved) -> str:
    """Claude phrases an answer strictly from the numbered sources."""
    import anthropic  # imported lazily so the app runs without the package
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sources = "\n\n".join(
        f"[{i + 1}] ({c.doc} — {c.section})\n{c.text}"
        for i, (c, _score) in enumerate(retrieved)
    )
    system = (
        "Answer the user's question using ONLY the numbered sources provided. "
        "Cite the sources you rely on inline, like [1] or [2]. "
        "If the answer is not contained in the sources, reply with exactly: "
        f'"{REFUSAL_TEXT}". Keep the answer to 1-3 sentences.'
    )
    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {query}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def generate_answer(query: str, retrieved: Retrieved) -> Tuple[str, str]:
    """Return (answer_text, mode). Falls back to extractive on any LLM error."""
    if ANTHROPIC_API_KEY:
        try:
            return llm_answer(query, retrieved), "llm"
        except Exception:
            return extractive_answer(retrieved), "extractive (llm-fallback)"
    return extractive_answer(retrieved), "extractive"
