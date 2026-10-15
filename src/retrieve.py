import json

import numpy as np

from . import config

_chunks = None
_embedder = None
_reranker = None
_bm25 = None
_chroma = None


def _all_chunks():
    global _chunks
    if _chunks is None:
        with open(config.CHUNKS_PATH) as f:
            _chunks = [json.loads(line) for line in f]
    return _chunks


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def _get_collection():
    global _chroma
    if _chroma is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _chroma = client.get_or_create_collection(
            config.COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _chroma


def vector_search(query, k=20):
    emb = _get_embedder().encode(
        [config.QUERY_PREFIX + query], normalize_embeddings=True
    ).tolist()
    res = _get_collection().query(query_embeddings=emb, n_results=k)
    return [
        {"id": i, "text": t, "source": m["source"], "heading": m["heading"]}
        for i, t, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]


def bm25_search(query, k=20):
    global _bm25
    if _bm25 is None:
        from rank_bm25 import BM25Okapi

        _bm25 = BM25Okapi([c["text"].lower().split() for c in _all_chunks()])
    scores = _bm25.get_scores(query.lower().split())
    order = np.argsort(scores)[::-1][:k]
    chunks = _all_chunks()
    return [chunks[i] for i in order]


def hybrid_search(query, k=20):
    scores = {}
    by_id = {}
    for lst in (vector_search(query, k), bm25_search(query, k)):
        for rank, c in enumerate(lst):
            scores[c["id"]] = scores.get(c["id"], 0.0) + 1.0 / (60 + rank)
            by_id[c["id"]] = c
    order = sorted(scores, key=scores.get, reverse=True)[:k]
    return [by_id[i] for i in order]


def rerank(query, chunks, top_n=5):
    scores = _get_reranker().predict([(query, c["text"]) for c in chunks])
    order = np.argsort(scores)[::-1][:top_n]
    return [chunks[i] for i in order]
