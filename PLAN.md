# Self-Correcting RAG: Execution Plan

## Instructions for the agent

Build this project phase by phase, in order. After each phase, stop and report what was built and the "Done when" result. Do not add features, abstractions, frameworks, or files that are not listed here. If something is ambiguous, pick the simplest option and note it in the report.

## Goal

Build a small RAG system over one document set and prove, with numbers, that each layer helps:

- **A. Naive RAG**: vector search only.
- **B. Solid RAG**: hybrid search + reranking.
- **C. Self-correcting RAG**: B + a loop that grades results, rewrites weak queries, and abstains when evidence is weak.

The deliverable is the comparison of A vs B vs C on a test set, not just the code.

## Fixed decisions (do not change)

| Area | Choice |
|---|---|
| Language | Python 3.11+, managed with `uv` |
| Corpus | One folder of Markdown docs (50-200 files) placed in `data/docs/`. The user will provide it; any public technical documentation works. |
| Embeddings | `sentence-transformers`, model `BAAI/bge-small-en-v1.5` (local, no API key) |
| Vector store | `chromadb` (persistent, local) |
| Keyword search | `rank_bm25` |
| Reranker | `sentence-transformers` CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenRouter API (OpenAI-compatible) via the `openai` SDK, key in `OPENROUTER_API_KEY`. Model names read from env vars `MODEL` (default `openai/gpt-4o-mini`) and `JUDGE_MODEL` (default `openai/gpt-4o`). Temperature 0. |
| Loop | Plain Python `for` loop. No LangGraph, no LangChain, no agent framework. |

## Project structure

```
data/docs/          # the corpus (provided by user)
data/eval.jsonl     # test set (Phase 2)
src/
  config.py         # env vars, constants
  index.py          # chunk + embed + store (Phase 1)
  retrieve.py       # vector, bm25, hybrid, rerank (Phase 1)
  generate.py       # answer from chunks (Phase 1)
  agent.py          # self-correcting loop (Phase 3)
  evaluate.py       # runs A/B/C on the test set (Phase 2, extended in 3)
results/            # eval outputs
README.md           # short write-up (Phase 4)
pyproject.toml
```

---

## Phase 1: Baseline RAG (A and B)

### 1.1 Chunking and indexing: `index.py`
- Read all `.md` files in `data/docs/`.
- Split by Markdown headings first; if a section is longer than ~1500 characters, split it further with ~200 characters of overlap.
- Each chunk stores: `id`, `text`, `source` (file path), `heading` (the heading path, e.g. "Setup > Install").
- Write all chunks to `data/chunks.jsonl`.
- Embed chunks and store them in a persistent Chroma collection.

### 1.2 Retrieval: `retrieve.py`
Implement exactly these functions:
- `vector_search(query, k=20)`: Chroma similarity search.
- `bm25_search(query, k=20)`: BM25 over `chunks.jsonl` (built in memory at startup).
- `hybrid_search(query, k=20)`: merge the two lists with Reciprocal Rank Fusion (`score = sum(1 / (60 + rank))`).
- `rerank(query, chunks, top_n=5)`: score with the cross-encoder, return the top 5.

Each function returns a list of chunk dicts.

### 1.3 Generation: `generate.py`
- `answer(question, chunks)`: one LLM call. The prompt tells the model to answer only from the provided chunks and to cite the `source` of each chunk it used. No other logic.

### 1.4 Two pipelines
- **Pipeline A (naive)**: `vector_search(top 5)` then `answer`.
- **Pipeline B (solid)**: `hybrid_search` then `rerank(top 5)` then `answer`.

**Done when:** a small script call like `python -m src.generate "your question"` works for both pipelines and prints an answer with sources.

---

## Phase 2: Evaluation harness

Build this before the self-correcting loop. Without it, the loop cannot be judged.

### 2.1 Test set: `data/eval.jsonl`
About 40 questions, one JSON object per line:

```json
{"id": "q01", "type": "answerable", "question": "...", "reference_answer": "...", "source": "path/to/doc.md"}
```

Composition:
- **25 answerable**: clear questions with a known answer in the docs.
- **8 ambiguous**: vague or underspecified wording (e.g. missing which component or version). `reference_answer` is the answer for the most likely interpretation.
- **7 unanswerable**: plausible questions on the same topic that the docs do NOT answer. `reference_answer` is `null`.

The agent may draft candidate questions from the docs, but the user reviews and edits them before this phase is considered done. Questions must be written from the user's point of view, not copied from doc text.

### 2.2 Runner and metrics: `evaluate.py`
- Run a given pipeline over every question and save raw outputs to `results/<pipeline>.jsonl` (question, answer, retrieved sources, number of LLM calls, latency).
- Score with an LLM judge (`JUDGE_MODEL`), one call per question, returning JSON:
  - For answerable/ambiguous: `correct` (true/false against `reference_answer`).
  - For unanswerable: `abstained` (true/false; did the answer say it cannot find this?).
- Also track `retrieval_hit`: does any retrieved chunk come from the `source` file? (answerable/ambiguous only.)
- Print one summary table:

| Metric | Meaning |
|---|---|
| Accuracy | % correct on answerable + ambiguous |
| Retrieval hit rate | % where the right source was retrieved |
| Abstention rate | % of unanswerable questions correctly refused |
| Hallucination rate | % of unanswerable questions that got a confident answer (= 100% minus abstention rate) |
| Avg LLM calls / latency | cost proxy |

**Done when:** `python -m src.evaluate --pipeline A` and `--pipeline B` both produce the summary table. Expect B to beat A on accuracy and retrieval hit rate. If it does not, inspect chunking before moving on.

---

## Phase 3: Self-correcting loop (C)

### 3.1 The loop: `agent.py`

```
chunks = retrieve(question)                      # Pipeline B retrieval
for attempt in 0..MAX_RETRIES (=2):
    verdict = grade(question, chunks)            # sufficient: true/false + reason
    if verdict.sufficient: break
    query = rewrite(question, verdict.reason, previous_queries)
    chunks = retrieve(query)
if not verdict.sufficient:
    return ABSTAIN_MESSAGE
return answer(question, chunks)
```

Implement exactly three small functions, each one LLM call:

- `grade(question, chunks)`: asks "Do these chunks contain enough information to answer the question?" Returns JSON `{"sufficient": bool, "reason": "..."}`. The prompt must instruct: when in doubt, mark insufficient only if key information is truly missing.
- `rewrite(question, reason, previous_queries)`: returns one improved search query. It may expand abbreviations, add missing context, or split off the missing sub-topic. It must not repeat a previous query.
- Abstain message: a fixed string, e.g. "I couldn't find this in the provided documents."

Constants (`MAX_RETRIES`, abstain message) live in `config.py`.

### 3.2 Logging
- Log each attempt (query used, verdict, reason) into the result record so failures can be inspected in `results/C.jsonl`.

### 3.3 Run the evaluation
- `python -m src.evaluate --pipeline C`
- Print a comparison table of A, B, C side by side using the Phase 2 metrics.

**Done when:** all three pipelines appear in one comparison table, and C's per-question logs show at least a few cases where a rewrite fixed a bad retrieval and a few where it abstained.

### 3.4 One tuning pass (allowed, once)
Look at C's errors only in two categories:
- Wrongly abstained on an answerable question (grader too strict).
- Answered an unanswerable question (grader too lenient).

Adjust the `grade` prompt once, rerun, and record both runs. Do not tune further.

---

## Phase 4: Write-up: `README.md`

Keep it under one page:
1. What the project is (2-3 sentences).
2. How to run it (install, index, evaluate).
3. The A/B/C comparison table.
4. Three concrete examples: one where B beat A, one where C's rewrite fixed a failure, one where C correctly abstained but A/B hallucinated.
5. Honest limitations: grader errors, extra cost and latency of C, size of the test set.

**Done when:** README exists and every number in it comes from `results/`.

---

## Out of scope (do not build)

- UI, API server, or Docker
- LangGraph / LangChain / LlamaIndex
- Multiple vector databases or embedding model comparisons
- Query decomposition into parallel sub-agents
- Answer-vs-source support verification step
- Caching, streaming, auth, deployment
- Any dataset or feature not listed above

These can be added later, one at a time, once the A/B/C comparison exists.
