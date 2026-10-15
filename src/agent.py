from . import config
from .generate import answer, chat, parse_json
from .retrieve import hybrid_search, rerank

GRADE_PROMPT = """Question: {question}

Chunks:
{context}

Do these chunks contain enough information to answer the question? Mark sufficient only if the key information needed is present; when in doubt, mark insufficient only if key information is truly missing. Reply with JSON only: {{"sufficient": true, "reason": "..."}}"""

REWRITE_PROMPT = """Question: {question}
Why the current results are insufficient: {reason}
Previous search queries (do not repeat any): {previous}

Write one improved search query to find the missing information. You may expand abbreviations, add missing context, or narrow to the missing sub-topic. Reply with the query only."""


def _retrieve(query):
    return rerank(query, hybrid_search(query, 20), 5)


def grade(question, chunks):
    context = "\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in chunks)
    raw = chat(
        config.MODEL,
        [{"role": "user", "content": GRADE_PROMPT.format(question=question, context=context)}],
    )
    return parse_json(raw)


def rewrite(question, reason, previous_queries):
    raw = chat(
        config.MODEL,
        [
            {
                "role": "user",
                "content": REWRITE_PROMPT.format(
                    question=question,
                    reason=reason,
                    previous=" | ".join(previous_queries),
                ),
            }
        ],
    )
    return raw.strip().strip('"')


def run(question):
    queries = [question]
    chunks = _retrieve(question)
    calls = 0
    attempts = []
    verdict = {}

    for attempt in range(config.MAX_RETRIES + 1):
        verdict = grade(question, chunks)
        calls += 1
        attempts.append(
            {
                "query": queries[-1],
                "sufficient": verdict.get("sufficient"),
                "reason": verdict.get("reason"),
            }
        )
        if verdict.get("sufficient") or attempt == config.MAX_RETRIES:
            break
        query = rewrite(question, verdict.get("reason", ""), queries)
        calls += 1
        queries.append(query)
        chunks = _retrieve(query)

    if not verdict.get("sufficient"):
        return {
            "answer": config.ABSTAIN_MESSAGE,
            "chunks": chunks,
            "llm_calls": calls,
            "attempts": attempts,
        }
    calls += 1
    return {
        "answer": answer(question, chunks),
        "chunks": chunks,
        "llm_calls": calls,
        "attempts": attempts,
    }
