import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
EVAL_PATH = ROOT / "data" / "eval.jsonl"
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION = "docs"

# bge-* retrieval models are trained with this instruction on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("MODEL", "openai/gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-4o")
TEMPERATURE = 0

MAX_RETRIES = 2
ABSTAIN_MESSAGE = "I couldn't find this in the provided documents."
