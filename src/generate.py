import argparse
import json
import os

from . import config

_client = None

ANSWER_PROMPT = """Answer the question using only the provided chunks. If the chunks do not contain the answer, say you cannot find it in the provided documents. Cite the source of every chunk you use.

Chunks:
{context}

Question: {question}"""


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
    return _client


def chat(model, messages):
    resp = _get_client().chat.completions.create(
        model=model, messages=messages, temperature=config.TEMPERATURE
    )
    return resp.choices[0].message.content


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])


def answer(question, chunks):
    context = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in chunks
    )
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    return chat(config.MODEL, [{"role": "user", "content": prompt}])


def pipeline_a(question):
    from .retrieve import vector_search

    chunks = vector_search(question, 5)
    return {"answer": answer(question, chunks), "chunks": chunks, "llm_calls": 1}


def pipeline_b(question):
    from .retrieve import hybrid_search, rerank

    chunks = rerank(question, hybrid_search(question, 20), 5)
    return {"answer": answer(question, chunks), "chunks": chunks, "llm_calls": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--pipeline", choices=["A", "B"], default="B")
    args = parser.parse_args()
    run = pipeline_a if args.pipeline == "A" else pipeline_b
    print(run(args.question)["answer"])


if __name__ == "__main__":
    main()
