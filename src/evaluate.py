import argparse
import json
import time

from . import agent, config
from .generate import chat, parse_json, pipeline_a, pipeline_b

PIPELINES = {"A": pipeline_a, "B": pipeline_b, "C": agent.run}

JUDGE_CORRECT = """You are grading a retrieval-augmented answer. Judge only the answer's factual content against the reference, ignoring style.

Question: {question}
Reference answer: {reference}
Model answer: {response}

Does the model answer convey the same key facts as the reference answer? Reply with JSON only: {{"correct": true}} or {{"correct": false}}"""

JUDGE_ABSTAIN = """A question was asked that the source documents do not answer. A model should say it cannot find the answer rather than give a confident guess.

Question: {question}
Model answer: {response}

Did the model correctly refuse / say it could not find the answer in the documents? Reply with JSON only: {{"abstained": true}} or {{"abstained": false}}"""


def _load_eval():
    with open(config.EVAL_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def judge(item, response):
    if item["type"] == "unanswerable":
        prompt = JUDGE_ABSTAIN.format(question=item["question"], response=response)
    else:
        prompt = JUDGE_CORRECT.format(
            question=item["question"],
            reference=item["reference_answer"],
            response=response,
        )
    return parse_json(chat(config.JUDGE_MODEL, [{"role": "user", "content": prompt}]))


def evaluate(pipeline_name):
    items = _load_eval()
    records = []
    for item in items:
        start = time.time()
        out = PIPELINES[pipeline_name](item["question"])
        latency = time.time() - start
        verdict = judge(item, out["answer"])
        sources = [c["source"] for c in out["chunks"]]
        rec = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "answer": out["answer"],
            "sources": sources,
            "llm_calls": out["llm_calls"],
            "latency": latency,
            **verdict,
        }
        if "attempts" in out:
            rec["attempts"] = out["attempts"]
        if item["type"] != "unanswerable":
            rec["retrieval_hit"] = item["source"] in sources
        records.append(rec)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / f"{pipeline_name}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def summarize(records):
    ans = [r for r in records if r["type"] != "unanswerable"]
    unans = [r for r in records if r["type"] == "unanswerable"]

    def rate(rows, key):
        return sum(r.get(key) is True for r in rows) / len(rows) if rows else 0.0

    return {
        "Accuracy": rate(ans, "correct"),
        "Retrieval hit rate": rate(ans, "retrieval_hit"),
        "Abstention rate": rate(unans, "abstained"),
        "Hallucination rate": 1 - rate(unans, "abstained"),
        "Avg LLM calls": sum(r["llm_calls"] for r in records) / len(records),
        "Avg latency (s)": sum(r["latency"] for r in records) / len(records),
    }


def print_table(names, summaries):
    metrics = list(summaries[names[0]])
    print("| Metric | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for m in metrics:
        cells = []
        for n in names:
            v = summaries[n][m]
            cells.append(f"{v:.2f}" if "latency" in m or "calls" in m else f"{100*v:.0f}%")
        print(f"| {m} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, nargs="+", choices=sorted(PIPELINES))
    args = parser.parse_args()
    summaries = {name: summarize(evaluate(name)) for name in args.pipeline}
    print_table(args.pipeline, summaries)


if __name__ == "__main__":
    main()
