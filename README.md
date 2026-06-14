# CiteRAG — Grounded Document Q&A (RAG that cites and refuses)

Ask questions over a set of documents and get an answer that **cites the exact
source it came from** — and that says *"I couldn't find that in the documents"*
instead of guessing when the answer isn't there.

Most RAG demos fail in one expensive way: when the retrieved context doesn't
contain the answer, the model invents one anyway, confidently and uncited.
CiteRAG is built so that can't happen by accident.

> Python · FastAPI · dependency-free TF-IDF retrieval (zero setup) ·
> Claude-optional answer generation · citations + refusal guardrail · eval tests.

---

## The design (the part that matters)

1. **Retrieve.** Documents are split into citable sections and indexed. A
   question retrieves the top-k most similar chunks
   ([`app/index.py`](app/index.py), [`app/corpus.py`](app/corpus.py)).
2. **Guardrail before generation.** If the best chunk isn't similar enough to
   the question, the system **refuses** — it never calls the generator with weak
   context ([`app/guardrail.py`](app/guardrail.py)). This is what stops
   hallucinated answers to out-of-scope questions.
3. **Grounded answer, always cited.** The answer is produced *only* from the
   retrieved chunks, and the response returns the sources behind it — document,
   section, snippet, and similarity score ([`app/service.py`](app/service.py)).
   Every answer is auditable back to the text it came from.

Answer generation has two interchangeable modes:

- **Extractive (default):** returns the most relevant sentences verbatim, cited.
  No API key, no network — and no possibility of fabrication.
- **LLM (optional):** with `ANTHROPIC_API_KEY` set, Claude phrases an answer
  **from the supplied sources only**, is told to cite inline, and to reply with
  the refusal text if the sources don't contain the answer.

The retrieval here is plain TF-IDF to keep the demo dependency-free and
transparent. Swapping in real embeddings (OpenAI, sentence-transformers) means
replacing one function — the guardrail, citation, and refusal logic are
unchanged.

---

## Run it

Requires Python 3.10+.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload     # serves http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000> and ask, or click a sample question. To enable
LLM-phrased answers, copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY`.

Drop your own `.md` files into `docs/` and restart — the index rebuilds from
whatever is there.

## Tests (the grounding is measurable)

```bash
python -m unittest discover -s tests -v
```

The eval checks both halves of the promise: in-scope questions are answered from
the correct document **with a citation**, and out-of-scope questions
(`"What is the company's annual revenue?"`, `"capital of France?"`) are
**refused** rather than answered with a confident guess.

## Example

| Question | Result |
|---|---|
| *What is the return window for items?* | "…within **30 days** of delivery…" — cited to *Returns & Refund Policy → Return window* |
| *Is there a restocking fee?* | "…a **15%** restocking fee…" — cited to *Returns → Restocking fee* |
| *What is the company's annual revenue?* | **Refused** — *"I couldn't find that in the documents."* |

## Project layout

```
app/
  config.py     env-driven settings (model, top-k, refusal threshold)
  text.py       tokenisation + light stemming (dependency-free)
  corpus.py     load markdown docs -> citable section chunks
  index.py      TF-IDF vector index + cosine retrieval
  guardrail.py  refuse-on-weak-retrieval logic
  generate.py   grounded answer (extractive, or Claude from sources only)
  service.py    orchestration + auditable {answer, refused, sources} envelope
  main.py       FastAPI app
docs/           the indexed corpus (sample Acme policy docs)
static/         single-page UI (answer + inline citations + sources)
tests/          grounding + refusal eval
```

---

*Demo project. The pattern — retrieve, refuse on weak evidence, answer only from
sources, cite everything — is the one I bring to production RAG so a system's
answers are trustworthy and auditable rather than confidently wrong.*
