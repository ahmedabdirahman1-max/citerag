"""FastAPI entrypoint.

Run:  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000

The document index is built from the markdown files in docs/ at startup.
"""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ANTHROPIC_API_KEY, ANSWER_MODEL, REFUSAL_THRESHOLD
from .corpus import load_chunks
from .index import TfidfIndex
from .service import answer_question, SAMPLE_QUESTIONS

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="CiteRAG — grounded document Q&A (demo)", version="1.0.0")

# Built once at startup and reused across requests.
_INDEX = TfidfIndex()


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def _build_index() -> None:
    _INDEX.fit(load_chunks())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks_indexed": len(_INDEX.chunks),
        "answer_mode": "llm" if ANTHROPIC_API_KEY else "extractive",
        "answer_model": ANSWER_MODEL if ANTHROPIC_API_KEY else None,
        "refusal_threshold": REFUSAL_THRESHOLD,
    }


@app.get("/api/samples")
def samples() -> dict:
    return {"samples": SAMPLE_QUESTIONS}


@app.post("/api/ask")
def ask(req: AskRequest) -> JSONResponse:
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Empty question."}, status_code=400)
    return JSONResponse(answer_question(question, _INDEX))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
