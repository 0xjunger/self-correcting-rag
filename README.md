# Self-Correcting RAG

A small RAG system over one folder of Markdown docs that compares three pipelines: **A** naive vector search, **B** hybrid search + reranking, and **C** B plus a self-correcting loop that grades retrieval, rewrites weak queries, and abstains when evidence is weak. The point is to show with numbers which layers help.

## How to run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The LLM layer uses OpenRouter's OpenAI-compatible API.

```bash
uv sync
export OPENROUTER_API_KEY=...        # optional: MODEL, JUDGE_MODEL
# put Markdown docs in data/docs/

python -m src.index                  # chunk + embed + store
python -m src.generate "your question" --pipeline A   # or B
python -m src.evaluate --pipeline A B C               # full comparison
```

`data/eval.jsonl` holds the test set (one JSON object per line: `id`, `type`, `question`, `reference_answer`, `source`). Raw outputs are written to `results/<pipeline>.jsonl`.

## A/B/C comparison

> Placeholder — populated from `results/` after the evaluation runs.

| Metric | A | B | C |
|---|---|---|---|
| Accuracy | — | — | — |
| Retrieval hit rate | — | — | — |
| Abstention rate | — | — | — |
| Hallucination rate | — | — | — |
| Avg LLM calls | — | — | — |
| Avg latency (s) | — | — | — |

## Examples

> Placeholders — each must come from `results/`.

- **B beat A:** _question, A's answer, B's answer._
- **C's rewrite fixed a failure:** _question, initial query, rewritten query, resulting answer._
- **C abstained where A/B hallucinated:** _question, A/B answer, C abstention._

## Limitations

- Grader errors: both the sufficiency grader and the LLM judge are model-based and can be wrong.
- C costs more: extra grade/rewrite calls and higher latency than A/B.
- Small test set; single corpus; no answer-vs-source verification step.
