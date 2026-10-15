import json
import re

import chromadb
from sentence_transformers import SentenceTransformer

from . import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(text):
    stack = []
    buf = []
    out = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append((" > ".join(t for _, t in stack), body))
        buf.clear()

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return out


def _split_long(text, size=1500, overlap=200):
    if len(text) <= size:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return parts


def build_chunks():
    chunks = []
    for path in sorted(config.DOCS_DIR.rglob("*.md")):
        rel = str(path.relative_to(config.DOCS_DIR))
        for heading, body in _sections(path.read_text(encoding="utf-8")):
            for piece in _split_long(body):
                chunks.append(
                    {
                        "id": f"{rel}:{len(chunks)}",
                        "text": piece,
                        "source": rel,
                        "heading": heading,
                    }
                )
    return chunks


def main():
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    if not chunks:
        raise SystemExit(f"No .md files found in {config.DOCS_DIR}")

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in chunks],
        embeddings=embeddings,
    )
    print(f"Indexed {len(chunks)} chunks from {config.DOCS_DIR}")


if __name__ == "__main__":
    main()
